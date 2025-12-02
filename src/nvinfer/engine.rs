use crate::error::Error;
use crate::model::Model;
use crate::{ExecutionTask, QuantizationKind};

use async_tensorrt::NetworkDefinitionCreationFlags;
use async_tensorrt::{Builder, Engine, ExecutionContext, Parser, Runtime, engine::TensorIoMode};
use futures::Stream;
use futures::StreamExt;
use std::marker::PhantomData;
use std::pin::pin;
use std::time::Duration;
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
use tokio::sync::mpsc;
use tokio::time;

#[derive(Debug, Clone)]
pub struct Config {
    pub max_streams: usize,
    pub cache_dir: PathBuf,
    pub quant_kind: QuantizationKind,
    pub queue_size: usize,
    pub timeout: f32,
}

pub struct Session<In, M>
where
    M: Model<In>,
{
    pub(crate) config: Config,
    pub(crate) model: Arc<M>,
    pub(crate) contexts: Mutex<Vec<ExecutionContext<'static>>>,
    _m: PhantomData<In>,
}

impl<In, M> Session<In, M>
where
    In: Send + 'static,
    M::Out: 'static,
    M: Model<In> + 'static,
{
    pub async fn new(mut model: M, config: Config) -> Result<Self, Error> {
        log::info!(
            "[nvinfer] Loading model from onnx: {}:",
            model.onnx_path().display()
        );

        std::fs::create_dir_all(&config.cache_dir)?;

        let serialized_engine_path = Self::create_path(&model, &config);

        let engine = if let Some(engine) = Self::try_load_engine(&serialized_engine_path).await {
            engine
        } else {
            Self::engine_build(&model, &serialized_engine_path).await?
        };

        let num = engine.num_io_tensors();
        for i in 0..num {
            let name = engine.io_tensor_name(i);
            let shape = engine.tensor_shape(&name);
            let io_mode = engine.tensor_io_mode(&name);

            log::info!("    {i} [{name}]: {io_mode:?} {shape:?}");

            match io_mode {
                TensorIoMode::None => (),
                TensorIoMode::Input => model.inputs_info().set(&name, &shape),
                TensorIoMode::Output => model.outputs_info().set(&name, &shape),
            }
        }

        Ok(Self {
            model: Arc::new(model),
            contexts: Mutex::new(
                ExecutionContext::from_engine_many(engine, config.max_streams).await?,
            ),
            config,
            _m: PhantomData,
        })
    }

    pub(crate) async fn try_load_engine(path: &Path) -> Option<Engine> {
        if path.exists() {
            let buffer = tokio::fs::read(path).await.ok()?;

            match Runtime::new().await.deserialize_engine(&buffer).await {
                Ok(engine) => Some(engine),
                Err(err) => {
                    log::warn!("failied to load serialized state: {err}!");
                    None
                }
            }
        } else {
            None
        }
    }

    pub(crate) async fn engine_build(
        model: &M,
        serialized_engine_path: &Path,
    ) -> Result<Engine, Error> {
        log::info!("building model...");

        let mut builder = Builder::new().await?;
        let mut builder_config = match model.quantization() {
            QuantizationKind::Float32 => builder.config().await,
            QuantizationKind::Float16 => builder.config().await.with_fp16(),
            _ => unimplemented!(),
        };

        let mut profile = builder.optimization_profile()?;

        for input in model.inputs() {
            profile.set_max_dimensions(input.name.as_ref(), input.shape_min.as_ref());
            profile.set_opt_dimensions(input.name.as_ref(), input.shape_opt.as_ref());
            profile.set_min_dimensions(input.name.as_ref(), input.shape_max.as_ref());
        }

        builder_config.add_optimization_profile(profile)?;

        let network_definition =
            builder.network_definition(NetworkDefinitionCreationFlags::ExplicitBatchSize);

        let mut network_definition =
            Parser::parse_network_definition_from_file(network_definition, &model.onnx_path())?;

        let plan = builder
            .build_serialized_network(&mut network_definition, builder_config)
            .await?;

        tokio::fs::write(serialized_engine_path, plan.as_bytes()).await?;

        Ok(Runtime::new()
            .await
            .deserialize_engine_from_plan(&plan)
            .await?)
    }

    pub async fn spawn_execution_task(
        &self,
        stream: impl Stream<Item = Result<In, Error>> + Send + 'static,
    ) -> Result<ExecutionTask<M::Out>, Error> {
        let context = self
            .contexts
            .lock()
            .unwrap()
            .pop()
            .ok_or(Error::NoExecutionContext)?;

        let mut infer_stream = super::stream::InferStream::new(context, self.model.clone()).await?;
        let timeout = Duration::from_secs_f32(self.config.timeout);
        let (tx, rx) = mpsc::channel(self.config.queue_size);
        let handler = tokio::spawn(async move {
            let mut stream = pin!(stream);
            let mut current_timeout = Duration::from_secs(u64::MAX);

            while let Some(res) = time::timeout(current_timeout, stream.next())
                .await
                .transpose()
            {
                match res {
                    // Got input
                    Ok(Ok(input)) => {
                        current_timeout = timeout;
                        if let Err(err) = infer_stream.infer(input, tx.clone()).await {
                            tx.send(Err(err)).await.unwrap();
                        }
                    }

                    // Timeout elapsed
                    Err(..) => {
                        current_timeout = Duration::from_secs(u64::MAX);
                        if let Err(err) = infer_stream.flush(tx.clone()).await {
                            tx.send(Err(err)).await.unwrap();
                        }
                    }

                    // Got an error
                    Ok(Err(err)) => tx.send(Err(err)).await.unwrap(),
                }
            }

            if let Err(err) = infer_stream.flush(tx.clone()).await {
                tx.send(Err(err)).await.unwrap();
            }
        });

        Ok(ExecutionTask { rx, handler })
    }

    pub(crate) fn create_path(model: &M, config: &Config) -> PathBuf {
        let mut path = config.cache_dir.clone();

        path.push(format!(
            "{}-{}-{}.trt",
            model.onnx_path().file_name().unwrap().to_str().unwrap(),
            model.quantization(),
            model.fingerprint()
        ));
        path
    }
}

impl<In, M> crate::Engine<In> for Session<In, M>
where
    M: Model<In> + 'static,
    M::Out: Send + 'static,
    In: Send + 'static,
{
    type Out = M::Out;

    async fn infer(
        &self,
        input: impl Stream<Item = Result<In, Error>> + Send + 'static,
    ) -> Result<impl Stream<Item = Result<Self::Out, Error>> + Send + 'static, Error> {
        self.spawn_execution_task(input).await
    }
}
