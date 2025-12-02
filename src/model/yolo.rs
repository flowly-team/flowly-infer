use std::{path::PathBuf, sync::Arc};

use async_cuda::HostBuffer;
use image::{
    Rgb32FImage,
    imageops::{self, FilterType},
};
use ndarray::{Array3, Axis};

use crate::{
    AsVal, Projection, QuantizationKind,
    bbox::BoundingBox,
    error::Error,
    model::{BindingInfo, IoBindingInfo},
};

use super::{IoBinding, IoDefinition, Model};

const MODEL_NAME: &str = "yolov8";
const INPUT_NAME: &str = "images";
const OUTPUT_NAME: &str = "output0";

#[derive(Debug, Clone)]
pub struct YoloModel {
    threshold: f32,
    onnx: PathBuf,
    iou_threshold: f32,
    inputs: [IoDefinition; 1],
    quantization: QuantizationKind,
    inputs_info: YoloInputsInfo,
    outputs_info: YoloOutputsInfo,
    batch_size: u32,
    height: u32,
    width: u32,
}

impl YoloModel {
    pub fn new<P: Into<PathBuf>>(
        threshold: f32,
        iou_threshold: f32,
        quantization: QuantizationKind,
        batch: u32,
        width: u32,
        height: u32,
        onnx: P,
    ) -> Self {
        Self {
            inputs: [IoDefinition::new_static_name(
                INPUT_NAME,
                &[batch as _, 3, height as _, width as _],
                &[batch as _, 3, height as _, width as _],
                &[batch as _, 3, height as _, width as _],
            )],
            batch_size: batch,
            threshold,
            iou_threshold,
            onnx: onnx.into(),
            width,
            height,
            quantization,
            inputs_info: Default::default(),
            outputs_info: Default::default(),
        }
    }
}

pub struct YoloInputs {
    images: HostBuffer<f32>,
}

pub struct YoloOutputs {
    output0: HostBuffer<f32>,
}

#[derive(Debug, Default, Clone)]
pub struct YoloInputsInfo {
    images: BindingInfo<4>,
}

impl IoBindingInfo for YoloInputsInfo {
    #[inline]
    fn shape<const N: usize>(&self, name: &str) -> [usize; N] {
        assert_eq!(name, INPUT_NAME);
        self.images.into()
    }

    #[inline]
    fn shape_dyn(&self, name: &str) -> &[usize] {
        assert_eq!(name, INPUT_NAME);
        &self.images.shape
    }

    #[inline]
    fn size(&self, name: &str) -> usize {
        assert_eq!(name, INPUT_NAME);
        self.images.size
    }

    #[inline]
    fn set(&mut self, name: &str, shape: &[usize]) {
        assert_eq!(name, INPUT_NAME);
        self.images.set(shape)
    }
}

#[derive(Debug, Default, Clone)]
pub struct YoloOutputsInfo {
    output0: BindingInfo<3>,
}

impl IoBindingInfo for YoloOutputsInfo {
    #[inline]
    fn shape<const N: usize>(&self, name: &str) -> [usize; N] {
        assert_eq!(name, OUTPUT_NAME);
        self.output0.into()
    }

    #[inline]
    fn shape_dyn(&self, name: &str) -> &[usize] {
        assert_eq!(name, OUTPUT_NAME);
        &self.output0.shape
    }

    #[inline]
    fn size(&self, name: &str) -> usize {
        assert_eq!(name, OUTPUT_NAME);
        self.output0.size
    }

    fn set(&mut self, name: &str, shape: &[usize]) {
        assert_eq!(name, OUTPUT_NAME);
        self.output0.set(shape)
    }
}

impl IoBinding<f32> for YoloInputs {
    #[inline]
    fn keys(&self) -> &[&str] {
        &[INPUT_NAME]
    }

    #[inline]
    fn count(&self) -> usize {
        1
    }

    #[inline]
    fn size(&self, name: &str) -> usize {
        assert_eq!(name, INPUT_NAME);
        self.images.num_elements()
    }

    #[inline]
    fn get(&self, name: &str) -> &[f32] {
        assert_eq!(name, INPUT_NAME);
        self.images.as_slice()
    }

    #[inline]
    fn get_mut(&mut self, name: &str) -> &mut [f32] {
        assert_eq!(name, INPUT_NAME);
        self.images.as_mut_slice()
    }

    #[inline]
    fn buffers(&self) -> impl Iterator<Item = (&str, &HostBuffer<f32>)> {
        [(INPUT_NAME, &self.images)].into_iter()
    }

    #[inline]
    fn buffers_mut(&mut self) -> impl Iterator<Item = (&str, &mut HostBuffer<f32>)> {
        [(INPUT_NAME, &mut self.images)].into_iter()
    }
}

impl IoBinding<f32> for YoloOutputs {
    #[inline]
    fn keys(&self) -> &[&str] {
        &[OUTPUT_NAME]
    }

    #[inline]
    fn count(&self) -> usize {
        1
    }

    #[inline]
    fn size(&self, name: &str) -> usize {
        assert_eq!(name, OUTPUT_NAME);
        self.output0.num_elements()
    }

    #[inline]
    fn get(&self, name: &str) -> &[f32] {
        assert_eq!(name, OUTPUT_NAME);
        self.output0.as_slice()
    }

    #[inline]
    fn get_mut(&mut self, name: &str) -> &mut [f32] {
        assert_eq!(name, OUTPUT_NAME);
        self.output0.as_mut_slice()
    }

    #[inline]
    fn buffers(&self) -> impl Iterator<Item = (&str, &HostBuffer<f32>)> {
        [(OUTPUT_NAME, &self.output0)].into_iter()
    }

    #[inline]
    fn buffers_mut(&mut self) -> impl Iterator<Item = (&str, &mut HostBuffer<f32>)> {
        [(OUTPUT_NAME, &mut self.output0)].into_iter()
    }
}

#[derive(Debug, Clone)]
pub struct YoloDetection<I> {
    pub input: Arc<I>,
    pub score: f32,
    pub class: u32,
    pub bbox: BoundingBox,
}

impl<I: AsRef<Rgb32FImage>> AsRef<Rgb32FImage> for YoloDetection<I> {
    fn as_ref(&self) -> &Rgb32FImage {
        self.input.as_ref().as_ref()
    }
}

impl<I> AsVal<Projection> for YoloDetection<I> {
    fn as_val(&self) -> Projection {
        self.bbox.projection()
    }
}

impl<I: AsRef<Rgb32FImage> + Send + Sync> Model<I> for YoloModel {
    type TIn = f32;
    type TOut = f32;
    type Out = YoloDetection<I>;
    type Inputs = YoloInputs;
    type Outputs = YoloOutputs;
    type InputsInfo = YoloInputsInfo;
    type OutputsInfo = YoloOutputsInfo;

    #[inline]
    fn name(&self) -> &str {
        MODEL_NAME
    }

    #[inline]
    fn inputs(&self) -> &[IoDefinition] {
        &self.inputs
    }

    #[inline]
    fn onnx_path(&self) -> &std::path::Path {
        &self.onnx
    }

    #[inline]
    fn batch_size(&self) -> usize {
        self.batch_size as _
    }

    #[inline]
    fn quantization(&self) -> QuantizationKind {
        self.quantization
    }

    #[inline]
    fn inputs_info(&mut self) -> &mut dyn IoBindingInfo {
        &mut self.inputs_info
    }

    #[inline]
    fn outputs_info(&mut self) -> &mut dyn IoBindingInfo {
        &mut self.outputs_info
    }

    #[inline]
    async fn create_inputs(&self) -> Self::Inputs {
        YoloInputs {
            images: HostBuffer::new(self.inputs_info.images.size).await,
        }
    }

    #[inline]
    async fn create_outputs(&self) -> Self::Outputs {
        YoloOutputs {
            output0: HostBuffer::new(self.outputs_info.output0.size).await,
        }
    }

    fn pre_process(&self, input: &I, index: usize, dst: &mut Self::Inputs) {
        let img = imageops::resize(
            input.as_ref(),
            self.width,
            self.height,
            FilterType::Triangle,
        );

        dst.view_mut(INPUT_NAME, self.inputs_info.images.shape)
            .index_axis_mut(Axis(0), index)
            .assign(
                &Array3::from_shape_vec(
                    [self.height as usize, self.width as usize, 3],
                    img.into_vec(),
                )
                .unwrap()
                .permuted_axes([2, 0, 1]),
            );
    }

    fn post_process(
        &self,
        input: I,
        index: usize,
        outputs: &Self::Outputs,
    ) -> impl futures::Stream<Item = Result<Self::Out, Error>> + Send {
        let mut boxes = Vec::new();
        let x_scale = input.as_ref().width() as f32 / self.width as f32;
        let y_scale = input.as_ref().height() as f32 / self.height as f32;

        async_stream::stream! {
            let input = Arc::new(input);
            let output0 = outputs.view(OUTPUT_NAME, self.outputs_info.output0.shape).index_axis_move(Axis(0), index);

            for row in output0.t().axis_iter(Axis(0)) {
                let (class, score) = row
                    .iter()
                    .skip(4)
                    .copied()
                    .enumerate()
                    .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
                    .unwrap();

                if score < self.threshold {
                    continue;
                }

                let xc = row[0usize] * x_scale;
                let yc = row[1usize] * y_scale;
                let hw = (row[2usize] * x_scale) / 2.0;
                let hh = (row[3usize] * y_scale) / 2.0;

                boxes.push(YoloDetection {
                    input: input.clone(),
                    score,
                    class: class as _,
                    bbox: BoundingBox {
                        x1: xc - hw,
                        y1: yc - hh,
                        x2: xc + hw,
                        y2: yc + hh,
                    },
                });
            }

            boxes.sort_by(|box1, box2| box1.score.total_cmp(&box2.score));

            while let Some(current) = boxes.pop() {
                boxes.retain(|x| current.class != x.class || current.bbox.iou(&x.bbox) < self.iou_threshold);
                yield Ok(current);
            }
        }
    }
}
