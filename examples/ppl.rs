#![allow(clippy::manual_retain)]

use std::path::PathBuf;

use async_cuda::Device;
use clap::Parser;
use flowly_infer::model::Either;
use flowly_infer::model::yolo::YoloModel;
use flowly_infer::model::{diffiqa::DiffIQAModel, retinaface::RetinafaceModel};
use flowly_infer::{Engine, InterpolationKind, QuantizationKind, nvinfer, nvinfer::nvinfer};
use futures::StreamExt;

#[derive(Debug, clap::Parser)]
struct Args {
    #[clap(short = 'c', default_value = "1000")]
    count: usize,

    #[clap(short = 's', default_value = "10")]
    streams: usize,

    #[clap(short = 'd', default_value = "yolo")]
    detector: String,

    #[clap(short = 'q', default_value = "fp16")]
    quantization: QuantizationKind,

    #[clap(short = 'b', default_value = "16")]
    detector_batch: usize,

    #[clap(short = 'a', default_value = "8")]
    scorer_batch: usize,

    #[clap()]
    input: PathBuf,
}

pub struct PipelineConfig {
    pub flush_timeout: f32,

    pub nvinfer_cache_dir: String,
    pub nvinfer_max_streams: usize,

    // detector configuration
    pub detector_kind: String,
    pub detector_threshold: f32,
    pub detector_batch: usize,
    pub detector_quantization: QuantizationKind,
    pub detector_onnx: String,
    pub detector_width: u32,
    pub detector_height: u32,
    pub detector_interpolation: InterpolationKind,
    pub detector_queue: usize,
    pub detector_engine: String,

    // scorer configuration
    pub scorer_kind: String,
    pub scorer_batch: usize,
    pub scorer_quantization: QuantizationKind,
    pub scorer_onnx: String,
    pub scorer_interpolation: InterpolationKind,
    pub scorer_queue: usize,
    pub scorer_engine: String,
}

#[derive(Clone)]
pub struct Input<T> {
    pub image: T,
    pub index: u32,
    pub stream: u32,
}

impl<T> AsRef<T> for Input<T> {
    fn as_ref(&self) -> &T {
        &self.image
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    let params = Args::parse();
    log::info!("Pipeline example config:\n {:#?}", params);

    let count = params.count;
    let streams = params.streams;
    let det_kind = params.detector;
    let image = params.input;
    let quantization = params.quantization;
    let detector_batch = params.detector_batch;
    let scorer_batch = params.scorer_batch;

    let original_img = image::open(image)?;
    let original_img_f32 = original_img.to_rgb32f();

    let mem_info = Device::memory_info().await?;
    println!("{:?}", mem_info);

    let yolo_detector = YoloModel::new(
        0.5,
        0.7,
        quantization,
        detector_batch as _,
        640,
        480,
        "./assets/model.onnx",
    );

    let retinaface_detector = RetinafaceModel::new(
        detector_batch as _,
        0.5,
        QuantizationKind::Float16,
        "./assets/retinaface_f32.onnx",
        640,
        480,
    );

    let detector = match det_kind.as_ref() {
        "yolo" => Either::Left(yolo_detector),
        "retinaface" => Either::Right(retinaface_detector),
        _ => unimplemented!(),
    };

    let scorer = DiffIQAModel::new(
        scorer_batch as _,
        QuantizationKind::Float32,
        "./assets/diffiqa.onnx",
    );

    let config = nvinfer::Config {
        max_streams: streams,
        cache_dir: "./trt_cache".into(),
        quant_kind: QuantizationKind::Float32,
        timeout: 1.0,
        queue_size: 8,
    };
    let ppl = (
        nvinfer(detector, config.clone()).await?,
        nvinfer(scorer, config).await?,
    );

    let mut stream_counter = 0u32;
    futures::stream::repeat_with(|| {
        let mut counter = 0u32;
        stream_counter += 1;

        let img = original_img_f32.clone();

        ppl.infer_stream(
            futures::stream::repeat_with(move || {
                let index = counter;
                counter += 1;

                Ok(Input {
                    stream: stream_counter,
                    index,
                    image: img.clone(),
                })
            })
            .take(count),
        )
    })
    .take(streams)
    .for_each_concurrent(None, async |fut| {
        let mut stream = match fut.await {
            Ok(ok) => ok,
            Err(err) => {
                log::error!("err: {err}");
                return;
            }
        };

        tokio::spawn(async move {
            let mut counter = 0;
            let time = std::time::Instant::now();

            while let Some(x) = stream.next().await {
                if let Ok(_val) = x {
                    // let s = val.inner.input.stream;
                    // let i = val.inner.input.index;
                    // let cx = val.inner.bbox.center_left() as u32;
                    // let cy = val.inner.bbox.center_top() as u32;

                    // println!("[{}/{}] score: {}", s, i, val.score);
                    // println!(
                    //     "[{}/{}] det score {} bbox: {:?}",
                    //     s, i, val.inner.score, val.inner.bbox
                    // );

                    // let mut dy: DynamicImage = val.inner.input.image.clone().into();
                    // dy.crop(
                    //     val.inner.bbox.left() as _,
                    //     val.inner.bbox.top() as _,
                    //     val.inner.bbox.width() as _,
                    //     val.inner.bbox.height() as _,
                    // )
                    // .to_rgb8()
                    // .save(format!("/tmp/img-{s}-{i}-{cx}:{cy}.jpg"))
                    // .unwrap();

                    counter += 1;
                }
            }

            let elapsed = time.elapsed().as_secs_f32();

            println!("[{}] fps: {}", counter, count as f32 / elapsed);
        })
        .await
        .unwrap();
    })
    .await;

    Ok(())
}
