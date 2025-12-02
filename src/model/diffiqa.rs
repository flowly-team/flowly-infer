use std::path::PathBuf;

use async_cuda::HostBuffer;
use image::Rgb32FImage;
use imageproc::geometric_transformations::{self, Interpolation};
use ndarray::{Array3, Axis};

use crate::{
    AsVal, Model, Projection, QuantizationKind,
    error::Error,
    model::{BindingInfo, IoBinding, IoBindingInfo, IoDefinition},
};

const WIDTH: usize = 112;
const HEIGHT: usize = 112;

#[derive(Debug, Clone)]
pub struct DiffIQAModel {
    onnx: PathBuf,
    inputs: [IoDefinition; 1],
    batch_size: u32,
    quantization: QuantizationKind,
    inputs_info: DiffIQAInputsInfo,
    outputs_info: DiffIQAOutputsInfo,
}

impl DiffIQAModel {
    pub fn new(batch_size: u32, quantization: QuantizationKind, path: impl Into<PathBuf>) -> Self {
        Self {
            onnx: path.into(),
            inputs: [IoDefinition::new_static_name(
                "input",
                &[batch_size as _, 3, HEIGHT as _, WIDTH as _],
                &[batch_size as _, 3, HEIGHT as _, WIDTH as _],
                &[batch_size as _, 3, HEIGHT as _, WIDTH as _],
            )],
            batch_size,
            quantization,
            inputs_info: Default::default(),
            outputs_info: Default::default(),
        }
    }
}

pub struct DiffIQAInputs {
    input: HostBuffer<f32>,
}

pub struct DiffIQAOutputs {
    output: HostBuffer<f32>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct DiffIQAInputsInfo {
    input: BindingInfo<4>,
}

impl IoBindingInfo for DiffIQAInputsInfo {
    #[inline]
    fn shape<const N: usize>(&self, name: &str) -> [usize; N] {
        assert_eq!(name, "input");
        self.input.into()
    }

    #[inline]
    fn shape_dyn(&self, name: &str) -> &[usize] {
        assert_eq!(name, "input");
        &self.input.shape
    }

    #[inline]
    fn size(&self, name: &str) -> usize {
        assert_eq!(name, "input");
        self.input.size
    }

    #[inline]
    fn set(&mut self, name: &str, shape: &[usize]) {
        assert_eq!(name, "input");
        self.input.set(shape);
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct DiffIQAOutputsInfo {
    output: BindingInfo<2>,
}

impl IoBindingInfo for DiffIQAOutputsInfo {
    #[inline]
    fn shape<const N: usize>(&self, name: &str) -> [usize; N] {
        assert_eq!(name, "output");
        self.output.into()
    }

    #[inline]
    fn shape_dyn(&self, name: &str) -> &[usize] {
        assert_eq!(name, "output");
        &self.output.shape
    }

    #[inline]
    fn size(&self, name: &str) -> usize {
        assert_eq!(name, "output");
        self.output.size
    }

    #[inline]
    fn set(&mut self, name: &str, shape: &[usize]) {
        assert_eq!(name, "output");
        self.output.set(shape);
    }
}

impl IoBinding<f32> for DiffIQAInputs {
    #[inline]
    fn keys(&self) -> &[&str] {
        &["input"]
    }

    #[inline]
    fn count(&self) -> usize {
        1
    }

    #[inline]
    fn size(&self, name: &str) -> usize {
        assert_eq!(name, "input");
        self.input.num_elements()
    }

    #[inline]
    fn get(&self, name: &str) -> &[f32] {
        assert_eq!(name, "input");
        self.input.as_slice()
    }

    #[inline]
    fn get_mut(&mut self, name: &str) -> &mut [f32] {
        assert_eq!(name, "input");
        self.input.as_mut_slice()
    }

    #[inline]
    fn buffers(&self) -> impl Iterator<Item = (&str, &HostBuffer<f32>)> {
        [("input", &self.input)].into_iter()
    }

    #[inline]
    fn buffers_mut(&mut self) -> impl Iterator<Item = (&str, &mut HostBuffer<f32>)> {
        [("input", &mut self.input)].into_iter()
    }
}

impl IoBinding<f32> for DiffIQAOutputs {
    #[inline]
    fn keys(&self) -> &[&str] {
        &["output"]
    }

    #[inline]
    fn count(&self) -> usize {
        1
    }

    #[inline]
    fn size(&self, name: &str) -> usize {
        assert_eq!(name, "output");
        self.output.num_elements()
    }

    #[inline]
    fn get(&self, name: &str) -> &[f32] {
        assert_eq!(name, "output");
        self.output.as_slice()
    }

    #[inline]
    fn get_mut(&mut self, name: &str) -> &mut [f32] {
        assert_eq!(name, "output");
        self.output.as_mut_slice()
    }

    #[inline]
    fn buffers(&self) -> impl Iterator<Item = (&str, &HostBuffer<f32>)> {
        [("output", &self.output)].into_iter()
    }

    #[inline]
    fn buffers_mut(&mut self) -> impl Iterator<Item = (&str, &mut HostBuffer<f32>)> {
        [("output", &mut self.output)].into_iter()
    }
}

#[derive(Debug, Clone)]
pub struct BestshotScore<I> {
    pub inner: I,
    pub score: f32,
}

impl<I> Model<I> for DiffIQAModel
where
    I: AsRef<image::Rgb32FImage> + AsVal<Projection> + Send,
{
    type TIn = f32;
    type TOut = f32;
    type Out = BestshotScore<I>;
    type Inputs = DiffIQAInputs;
    type Outputs = DiffIQAOutputs;
    type InputsInfo = DiffIQAInputsInfo;
    type OutputsInfo = DiffIQAOutputsInfo;

    #[inline]
    fn name(&self) -> &str {
        "diffiqa"
    }

    #[inline]
    fn inputs(&self) -> &[IoDefinition] {
        &self.inputs
    }

    #[inline]
    fn quantization(&self) -> QuantizationKind {
        self.quantization
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
    fn inputs_info(&mut self) -> &mut dyn IoBindingInfo {
        &mut self.inputs_info
    }

    #[inline]
    fn outputs_info(&mut self) -> &mut dyn IoBindingInfo {
        &mut self.outputs_info
    }

    #[inline]
    async fn create_inputs(&self) -> Self::Inputs {
        DiffIQAInputs {
            input: HostBuffer::new(self.inputs_info.input.size).await,
        }
    }

    #[inline]
    async fn create_outputs(&self) -> Self::Outputs {
        DiffIQAOutputs {
            output: HostBuffer::new(self.outputs_info.output.size).await,
        }
    }

    fn pre_process(&self, input: &I, index: usize, dst: &mut Self::Inputs) {
        let mut dst_input = dst
            .view_mut("input", self.inputs_info.input.shape)
            .index_axis_move(Axis(0), index);

        let m = input.as_val().m;
        let mut img = Rgb32FImage::new(WIDTH as _, HEIGHT as _);

        #[rustfmt::skip]
        let proj = geometric_transformations::Projection::from_matrix([
            m.m11, m.m12, m.m13,
            m.m21, m.m22, m.m23,
            m.m31, m.m32, m.m33,
        ]).unwrap();

        imageproc::geometric_transformations::warp_into(
            input.as_ref(),
            &proj,
            Interpolation::Bilinear,
            image::Rgb([0.0, 0.0, 0.0]),
            &mut img,
        );

        dst_input.assign(
            &Array3::from_shape_vec([HEIGHT, WIDTH, 3], img.into_vec())
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
        futures::stream::iter([Ok(BestshotScore {
            inner: input,
            score: outputs.view("output", self.outputs_info.output.shape)[[index, 0]],
        })])
    }
}
