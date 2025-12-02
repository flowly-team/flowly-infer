#![allow(clippy::manual_retain)]

use std::time::Duration;

use async_cuda::Device;
use futures::StreamExt;
use nvinfer::{
    Engine, QuantizationKind,
    model::{diffiqa::DiffIQAModel, yolo::YoloModel},
    nvinfer::nvinfer,
};

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
    let mut args = std::env::args();
    args.next().unwrap();

    let count = 10;
    let streams = 1;

    let original_img = image::open("./output.jpg")?;
    // let original_img_u8 = original_img.to_rgb8();
    let original_img_f32 = original_img.to_rgb32f();

    // let (img_width, img_height) = (original_img.width(), original_img.height());

    let mem_info = Device::memory_info().await?;
    println!("{:?}", mem_info);

    // let detector = insightface::InsightFaceModel::new(0.5, 0.7, "./det_10g.onnx");
    let detector = YoloModel::new(
        0.5,
        0.7,
        QuantizationKind::Float16,
        4,
        640,
        480,
        "./model.onnx",
    );

    let scorer = DiffIQAModel::new(8, QuantizationKind::Float32, "./diffiqa.onnx");
    let config = nvinfer::nvinfer::Config {
        max_streams: streams,
        cache_dir: "./trt_cache".into(),
        quant_kind: QuantizationKind::Float32,
        timeout: 1.0,
    };

    let ppl = (
        nvinfer(detector, config.clone()).await?,
        nvinfer(scorer, config).await?,
    );

    let mut stream_counter = 0u32;
    futures::stream::repeat_with(|| {
        stream_counter += 1;

        let img = original_img_f32.clone();

        ppl.infer(async_stream::stream! {
            for index in 0..count {
                yield Ok(Input {
                    stream: stream_counter,
                    index,
                    image: img.clone(),
                });

                tokio::time::sleep(Duration::from_secs_f32(0.5)).await;
            }
        })
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
                if let Ok(val) = x {
                    let s = val.inner.input.stream;
                    let i = val.inner.input.index;

                    println!("[{}/{}] score: {}", s, i, val.score);
                    println!(
                        "[{}/{}] det score {} bbox: {:?}",
                        s, i, val.inner.score, val.inner.bbox
                    );
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
