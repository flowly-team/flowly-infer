use std::sync::Arc;
use std::{ops::Mul, path::PathBuf};

use image::{
    Rgb32FImage,
    imageops::{self, FilterType},
};
use nalgebra::{ArrayStorage, Matrix, Matrix1x2, Matrix2, Matrix2x1, Matrix3};
use ndarray::{Array2, Array4, ArrayView2};

use crate::model::QuantizationKind;
use crate::{AsVal, Projection, bbox::BoundingBox, error::Error};

use super::{IoBinding, IoBindingInfo, IoBindingInfoBuilder, IoDefinition, Model};

const WIDTH: usize = 640;
const HEIGHT: usize = 640;
const STRIDES: [usize; 3] = [8, 16, 32];
const STRIDE_COUNT: [usize; 3] = [12800, 3200, 800];
const ARCFACE_DST: [(f32, f32); 5] = [
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
];

pub struct InsightFaceModel {
    threshold: f32,
    iou_threshold: f32,
    path: PathBuf,
    inputs: [IoDefinition; 1],
    quantization: QuantizationKind,
}

impl InsightFaceModel {
    pub fn new<P: Into<PathBuf>>(
        threshold: f32,
        iou_threshold: f32,
        quantization: QuantizationKind,
        path: P,
    ) -> Self {
        Self {
            threshold,
            iou_threshold,
            path: path.into(),
            inputs: [IoDefinition::new(
                "input.1",
                &[1, 3, HEIGHT as _, WIDTH as _],
                &[1, 3, HEIGHT as _, WIDTH as _],
                &[1, 3, HEIGHT as _, WIDTH as _],
            )],
            quantization,
        }
    }
}

#[derive(Debug, Default)]
pub struct InsightFaceInputs {
    input_1: ndarray::Array4<f32>,
}

#[derive(Debug, Default)]
pub struct InsightFaceOutputs {
    pub(crate) scores: [Array2<f32>; 3],
    pub(crate) bboxes: [Array2<f32>; 3],
    pub(crate) kpsses: [Array2<f32>; 3],
}

#[derive(Clone, Copy)]
pub struct InsightFaceInputsInfo {
    pub(crate) input_1: [usize; 4],
}

#[derive(Clone, Copy)]
pub struct InsightFaceOutputsInfo {
    pub(crate) scores: [[usize; 2]; 3],
    pub(crate) bboxes: [[usize; 2]; 3],
    pub(crate) kpsses: [[usize; 2]; 3],
}

impl IoBindingInfo for InsightFaceInputsInfo {
    fn len(&self) -> usize {
        1
    }

    fn shapes(&self) -> impl Iterator<Item = (&str, &[usize])> + Send {
        [("input.1", self.input_1.as_slice())].into_iter()
    }
}

impl IoBindingInfo for InsightFaceOutputsInfo {
    fn len(&self) -> usize {
        9
    }

    fn shapes(&self) -> impl Iterator<Item = (&str, &[usize])> + Send {
        [
            ("448", self.scores[0].as_slice()),
            ("471", self.scores[1].as_slice()),
            ("494", self.scores[2].as_slice()),
            ("451", self.bboxes[0].as_slice()),
            ("474", self.bboxes[1].as_slice()),
            ("497", self.bboxes[2].as_slice()),
            ("454", self.kpsses[0].as_slice()),
            ("477", self.kpsses[1].as_slice()),
            ("500", self.kpsses[2].as_slice()),
        ]
        .into_iter()
    }
}

impl IoBinding<f32> for InsightFaceInputs {
    type Info = InsightFaceInputsInfo;

    fn set(&mut self, name: &str, value: Vec<f32>, shape: &[usize]) {
        assert_eq!(name, "input.1");
        self.input_1 =
            Array4::from_shape_vec([shape[0], shape[1], shape[2], shape[3]], value).unwrap();
    }

    fn get(&self, name: &str) -> ndarray::ArrayViewD<'_, f32> {
        assert_eq!(name, "input.1");
        self.input_1.view().into_dyn()
    }
}

impl IoBinding<f32> for InsightFaceOutputs {
    type Info = InsightFaceOutputsInfo;

    fn set(&mut self, name: &str, value: Vec<f32>, shape: &[usize]) {
        match name {
            "448" => self.scores[0] = Array2::from_shape_vec([shape[0], shape[1]], value).unwrap(),
            "471" => self.scores[1] = Array2::from_shape_vec([shape[0], shape[1]], value).unwrap(),
            "494" => self.scores[2] = Array2::from_shape_vec([shape[0], shape[1]], value).unwrap(),
            "451" => self.bboxes[0] = Array2::from_shape_vec([shape[0], shape[1]], value).unwrap(),
            "474" => self.bboxes[1] = Array2::from_shape_vec([shape[0], shape[1]], value).unwrap(),
            "497" => self.bboxes[2] = Array2::from_shape_vec([shape[0], shape[1]], value).unwrap(),
            "454" => self.kpsses[0] = Array2::from_shape_vec([shape[0], shape[1]], value).unwrap(),
            "477" => self.kpsses[1] = Array2::from_shape_vec([shape[0], shape[1]], value).unwrap(),
            "500" => self.kpsses[2] = Array2::from_shape_vec([shape[0], shape[1]], value).unwrap(),
            val => unreachable!("{}", val),
        };
    }

    fn get(&self, name: &str) -> ndarray::ArrayViewD<'_, f32> {
        match name {
            "448" => self.scores[0].view().into_dyn(),
            "471" => self.scores[1].view().into_dyn(),
            "494" => self.scores[2].view().into_dyn(),
            "451" => self.bboxes[0].view().into_dyn(),
            "474" => self.bboxes[1].view().into_dyn(),
            "497" => self.bboxes[2].view().into_dyn(),
            "454" => self.kpsses[0].view().into_dyn(),
            "477" => self.kpsses[1].view().into_dyn(),
            "500" => self.kpsses[2].view().into_dyn(),
            val => unreachable!("{}", val),
        }
    }
}

#[derive(Default)]
pub struct InsightFaceInputsInfoBuilder {
    pub(crate) shape: Option<Vec<usize>>,
}

impl<'a> IoBindingInfoBuilder<'a, f32, InsightFaceInputs> for InsightFaceInputsInfoBuilder {
    type Error = Error;

    fn set_shape(&mut self, name: &str, shape: Vec<usize>) {
        assert_eq!(name, "input.1");
        self.shape = Some(shape);
    }

    fn build(self) -> Result<InsightFaceInputsInfo, Self::Error> {
        let shape = self.shape.unwrap();

        Ok(InsightFaceInputsInfo {
            input_1: [shape[0], shape[1], shape[2], shape[3]],
        })
    }
}

#[derive(Default)]
pub struct InsightFaceOutputsInfoBuilder {
    pub(crate) scores: [[usize; 2]; 3],
    pub(crate) bboxes: [[usize; 2]; 3],
    pub(crate) kpsses: [[usize; 2]; 3],
}

impl<'a> IoBindingInfoBuilder<'a, f32, InsightFaceOutputs> for InsightFaceOutputsInfoBuilder {
    type Error = Error;

    fn set_shape(&mut self, name: &str, shape: Vec<usize>) {
        match name {
            "448" => self.scores[0] = [shape[0], shape[1]],
            "471" => self.scores[1] = [shape[0], shape[1]],
            "494" => self.scores[2] = [shape[0], shape[1]],

            "451" => self.bboxes[0] = [shape[0], shape[1]],
            "474" => self.bboxes[1] = [shape[0], shape[1]],
            "497" => self.bboxes[2] = [shape[0], shape[1]],

            "454" => self.kpsses[0] = [shape[0], shape[1]],
            "477" => self.kpsses[1] = [shape[0], shape[1]],
            "500" => self.kpsses[2] = [shape[0], shape[1]],
            val => unreachable!("{}", val),
        }
    }

    fn build(self) -> Result<InsightFaceOutputsInfo, Self::Error> {
        Ok(InsightFaceOutputsInfo {
            scores: self.scores,
            bboxes: self.bboxes,
            kpsses: self.kpsses,
        })
    }
}

#[derive(Debug, Clone)]
pub struct InsightFaceDetection<I> {
    pub input: Arc<I>,
    pub score: f32,
    pub bbox: BoundingBox,
    pub keypts: [(f32, f32); 5],
}

impl<I> AsVal<Projection> for InsightFaceDetection<I> {
    fn as_val(&self) -> Projection {
        Projection {
            m: umeyama(&self.keypts, &ARCFACE_DST),
        }
    }
}

impl<I: AsRef<Rgb32FImage>> AsRef<Rgb32FImage> for InsightFaceDetection<I> {
    fn as_ref(&self) -> &Rgb32FImage {
        self.input.as_ref().as_ref()
    }
}

impl<I: Send + Sync + AsRef<Rgb32FImage>> Model<f32, f32, I> for InsightFaceModel {
    type Out = InsightFaceDetection<I>;
    type Inputs = InsightFaceInputs;
    type Outputs = InsightFaceOutputs;

    fn inputs(&self) -> &[IoDefinition] {
        &self.inputs
    }

    fn name(&self) -> &str {
        "insightface"
    }

    fn quantization(&self) -> QuantizationKind {
        self.quantization
    }

    fn inputs_info_builder(&self) -> impl IoBindingInfoBuilder<'_, f32, Self::Inputs> {
        InsightFaceInputsInfoBuilder::default()
    }
    fn outputs_info_builder(&self) -> impl IoBindingInfoBuilder<'_, f32, Self::Outputs> {
        InsightFaceOutputsInfoBuilder::default()
    }

    fn pre_process(&self, input: &I) -> Self::Inputs {
        let model_ratio = HEIGHT as f32 / WIDTH as f32;
        let input_width = input.as_ref().width() as f32;
        let input_height = input.as_ref().height() as f32;
        let im_ratio = input_height / input_width;

        let (new_width, new_height) = if im_ratio > model_ratio {
            (HEIGHT as u32, (HEIGHT as f32 / im_ratio) as u32)
        } else {
            (WIDTH as u32, (WIDTH as f32 * im_ratio) as u32)
        };

        let img = imageops::resize(input.as_ref(), new_width, new_height, FilterType::Triangle);

        let mut input_1 = Array4::zeros([1, 3, HEIGHT, WIDTH]);
        for out_row in 0..WIDTH {
            for out_col in 0..HEIGHT {
                if let Some(pixel) = img.get_pixel_checked(out_row as _, out_col as _) {
                    input_1[[0, 0, out_col, out_row]] = (pixel.0[0] - 0.5) * 2.0;
                    input_1[[0, 1, out_col, out_row]] = (pixel.0[1] - 0.5) * 2.0;
                    input_1[[0, 2, out_col, out_row]] = (pixel.0[2] - 0.5) * 2.0;
                }
            }
        }

        InsightFaceInputs { input_1 }
    }

    fn post_process(
        &self,
        input: I,
        outputs: Self::Outputs,
    ) -> impl futures::Stream<Item = Result<Self::Out, Error>> + Send {
        let orig_width = input.as_ref().width() as f32;
        let orig_height = input.as_ref().height() as f32;
        let ratio = if orig_width / orig_height > 1. {
            orig_width / 640.0
        } else {
            orig_height / 640.0
        };

        async_stream::stream! {
            let input = Arc::new(input);
            let mut faces = Vec::new();

            for (s_idx, stride) in STRIDES.into_iter().enumerate() {
                for index in 0..STRIDE_COUNT[s_idx] {
                    let score = outputs.scores[s_idx][[index, 0]];

                    if score > self.threshold {
                        faces.push(InsightFaceDetection {
                            score,
                            bbox: distance2bbox(index, stride, outputs.bboxes[s_idx].view()),
                            keypts: distance2kps(index, stride, outputs.kpsses[s_idx].view()),
                            input: input.clone()
                        });
                    }
                }
            }

            faces.sort_by(|a, b| a.score.total_cmp(&b.score));

            while let Some(mut current) = faces.pop() {
                faces.retain(|x| current.bbox.iou(&x.bbox) < self.iou_threshold);

                current.bbox = current.bbox.map(|el| el * ratio);
                current.keypts = current.keypts.map(|(x, y)| (x * ratio, y * ratio));

                yield Ok(current);
            }
        }
    }

    #[inline]
    fn onnx_path(&self) -> &std::path::Path {
        &self.path
    }
}

#[inline]
fn distance2bbox(index: usize, stride: usize, distance: ArrayView2<f32>) -> BoundingBox {
    let m = WIDTH / stride;
    let x = ((index / 2) * stride) % WIDTH;
    let y = (((index / 2) / m) * stride) % HEIGHT;

    let x1 = x as f32 - distance[[index, 0]] * stride as f32;
    let y1 = y as f32 - distance[[index, 1]] * stride as f32;

    let x2 = x as f32 + distance[[index, 2]] * stride as f32;
    let y2 = y as f32 + distance[[index, 3]] * stride as f32;

    BoundingBox::new(x1, y1, x2, y2)
}

#[inline]
fn distance2kps(index: usize, stride: usize, distance: ArrayView2<f32>) -> [(f32, f32); 5] {
    let m = WIDTH / stride;
    let x = ((index / 2) * stride) % WIDTH;
    let y = (((index / 2) / m) * stride) % HEIGHT;

    let x1 = x as f32 + distance[[index, 0]] * stride as f32;
    let y1 = y as f32 + distance[[index, 1]] * stride as f32;

    let x2 = x as f32 + distance[[index, 2]] * stride as f32;
    let y2 = y as f32 + distance[[index, 3]] * stride as f32;

    let x3 = x as f32 + distance[[index, 4]] * stride as f32;
    let y3 = y as f32 + distance[[index, 5]] * stride as f32;

    let x4 = x as f32 + distance[[index, 6]] * stride as f32;
    let y4 = y as f32 + distance[[index, 7]] * stride as f32;

    let x5 = x as f32 + distance[[index, 8]] * stride as f32;
    let y5 = y as f32 + distance[[index, 9]] * stride as f32;

    [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)]
}

/// Алгоритм `Кабша-Умеямы` - это метод нахождения оптимального перемещения, поворота
/// и масштабирования, который выравнивает два набора точек с минимальным среднеквадратичным отклонением (RMSD).
pub fn umeyama<const R: usize>(src: &[(f32, f32); R], dst: &[(f32, f32); R]) -> Matrix3<f32> {
    let src_x_sum: f32 = src.iter().map(|v| v.0).sum();
    let src_x_mean = src_x_sum / (R as f32);

    let src_y_sum: f32 = src.iter().map(|v| v.1).sum();
    let src_y_mean = src_y_sum / (R as f32);

    let dst_x_sum: f32 = dst.iter().map(|v| v.0).sum();
    let dst_x_mean = dst_x_sum / (R as f32);

    let dst_y_sum: f32 = dst.iter().map(|v| v.1).sum();
    let dst_y_mean = dst_y_sum / (R as f32);

    let src_demean_s = ArrayStorage(src.map(|v| [v.0 - src_x_mean, v.1 - src_y_mean]));
    let dst_demean_s = ArrayStorage(dst.map(|v| [v.0 - dst_x_mean, v.1 - dst_y_mean]));

    let src_demean = Matrix::from_array_storage(src_demean_s);
    let dst_demean = Matrix::from_array_storage(dst_demean_s);

    let a = std::ops::Mul::mul(dst_demean, &src_demean.transpose()) / (R as f32);
    let svd = Matrix::svd(a, true, true);

    let determinant = a.determinant();

    let mut d = [1f32; 2];

    if determinant < 0.0f32 {
        d[2 - 1] = -1.0f32;
    }

    let mut t = Matrix2::<f32>::identity();
    let s = svd.singular_values;
    let u = svd.u.unwrap();
    let v = svd.v_t.unwrap();

    let rank = a.rank(0.00001f32);

    if rank == 0 {
        panic!("Matrix rank is 0.");
    } else if rank == 2 - 1 {
        if u.determinant() * v.determinant() > 0.0 {
            u.mul_to(&v, &mut t);
        } else {
            let s = d[2 - 1];
            d[2 - 1] = -1f32;
            let dg = Matrix2::<f32>::new(d[0], 0f32, 0f32, d[1]);

            let udg = u.mul(&dg);
            udg.mul_to(&v, &mut t);
            d[2 - 1] = s;
        }
    } else {
        let dg = Matrix2::<f32>::new(d[0], 0f32, 0f32, d[1]);
        let udg = u.mul(&dg);
        udg.mul_to(&v, &mut t);
    }

    let ddd = Matrix1x2::new(d[0], d[1]);
    let d_x_s = ddd.mul(s);

    let var0 = src_demean.remove_row(0).variance();
    let var1 = src_demean.remove_row(1).variance();

    let varsum = var0 + var1;

    let scale = d_x_s.get((0, 0)).unwrap() / varsum;

    let dst_mean = Matrix2x1::<f32>::new(dst_x_mean, dst_y_mean);
    let src_mean = Matrix2x1::<f32>::new(src_x_mean, src_y_mean);
    let t_x_srcmean = t.mul(&src_mean);

    let xxx = scale * t_x_srcmean;
    let yyy = dst_mean - xxx;

    let m13 = *yyy.get(0).unwrap();
    let m23 = *yyy.get(1).unwrap();

    let m00x22 = t * scale;

    let m11 = m00x22.m11;
    let m21 = m00x22.m21;
    let m12 = m00x22.m12;
    let m22 = m00x22.m22;

    Matrix3::<f32>::new(m11, m12, m13, m21, m22, m23, 0f32, 0f32, 1f32)
}
