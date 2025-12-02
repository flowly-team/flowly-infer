use crate::error::Error;
use crate::model::IoBinding;

use std::collections::HashMap;
use std::pin::pin;
use std::sync::Arc;

use async_cuda::{DeviceBuffer, DynDeviceBuffer};
use async_tensorrt::ExecutionContext;
use futures::{Stream, StreamExt, stream};
use tokio::sync::mpsc;

use crate::model::Model;

pub struct InferStream<'a, I, O, In, M>
where
    I: Copy + 'static,
    O: Copy + 'static,
    M: Model<In, TIn = I, TOut = O>,
{
    pub(crate) context: ExecutionContext<'a>,
    pub(crate) stream: async_cuda::Stream,
    pub(crate) io_device_buffers: HashMap<String, DynDeviceBuffer>,

    pub(crate) model: Arc<M>,
    pub(crate) inputs: M::Inputs,
    pub(crate) outputs: M::Outputs,
    pub(crate) batch: Vec<In>,
}

impl<'a, I, O, In, M> InferStream<'a, I, O, In, M>
where
    I: Copy + 'static,
    O: Copy + 'static,
    M: Model<In, TIn = I, TOut = O>,
{
    pub async fn new(context: ExecutionContext<'a>, model: Arc<M>) -> Result<Self, Error> {
        let batch_size = model.batch_size();
        let stream = async_cuda::Stream::new().await?;
        let inputs = model.create_inputs().await;
        let outputs = model.create_outputs().await;

        let mut io_device_buffers = HashMap::with_capacity(inputs.count() + outputs.count());

        for key in inputs.keys() {
            io_device_buffers.insert(
                key.to_string(),
                DeviceBuffer::<I>::new(inputs.size(key.as_ref()), &stream)
                    .await
                    .into_dyn(),
            );
        }

        for key in outputs.keys() {
            io_device_buffers.insert(
                key.to_string(),
                DeviceBuffer::<O>::new(outputs.size(key.as_ref()), &stream)
                    .await
                    .into_dyn(),
            );
        }

        Ok(Self {
            context,
            model,
            stream,
            io_device_buffers,
            inputs,
            outputs,
            batch: Vec::with_capacity(batch_size),
        })
    }

    async fn infer_inner(&mut self) -> Result<impl Stream<Item = Result<M::Out, Error>>, Error> {
        for (key, buff) in self.inputs.buffers() {
            self.io_device_buffers
                .get_mut(key)
                .unwrap()
                .copy_from(buff, &self.stream)
                .await?;
        }

        self.context
            .enqueue(&mut self.io_device_buffers, &self.stream)
            .await?;

        for (key, buff) in self.outputs.buffers_mut() {
            self.io_device_buffers
                .get_mut(key)
                .unwrap()
                .copy_to(buff, &self.stream)
                .await?;
        }

        let stream = stream::iter(self.batch.drain(..).enumerate())
            .map(|(idx, input)| {
                tokio::task::block_in_place(|| self.model.post_process(input, idx, &self.outputs))
            })
            .flatten();

        Ok(stream)
    }

    pub async fn infer(
        &mut self,
        input: In,
        sender: mpsc::Sender<Result<M::Out, Error>>,
    ) -> Result<(), Error> {
        tokio::task::block_in_place(|| {
            self.model
                .pre_process(&input, self.batch.len(), &mut self.inputs)
        });

        self.batch.push(input);

        if self.batch.len() == self.batch.capacity() {
            let mut stream = pin!(self.infer_inner().await?);

            while let Some(item) = stream.next().await {
                sender.send(item).await.unwrap();
            }
        }

        Ok(())
    }

    pub async fn flush(
        &mut self,
        sender: mpsc::Sender<Result<M::Out, Error>>,
    ) -> Result<(), Error> {
        if !self.batch.is_empty() {
            let mut stream = pin!(self.infer_inner().await?);

            while let Some(item) = stream.next().await {
                sender.send(item).await.unwrap();
            }
        }

        Ok(())
    }
}
