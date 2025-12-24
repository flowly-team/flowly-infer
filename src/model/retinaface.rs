use std::{path::PathBuf, sync::Arc};

use async_cuda::HostBuffer;
use image::{
    Rgb32FImage,
    imageops::{self, FilterType},
};
use itertools::Itertools;
use nalgebra::Point2;
use ndarray::{Array2, Array3, Axis};
use num::{Num, ToPrimitive};

const INPUT_NAME: &str = "input";
const OUTPUT_NAME_BBOXES: &str = "bboxes";
const OUTPUT_NAME_PROBS: &str = "probs";
const OUTPUT_NAME_LANDMARKS: &str = "landmarks";

const PRIORS: [[u32; 2]; 3] = [[16, 32], [64, 128], [256, 512]];
const STRIDES: [u32; 3] = [8, 16, 32];
const SCALES: [[f32; 2]; 3] = [[2.0, 1.0], [8.0, 4.0], [32.0, 16.0]];
const VARS: [f32; 2] = [0.1, 0.2];
const ANCHORS_FPN: [[[f32; 4]; 2]; 3] = [
    [
        [-248.0f32, -248.0, 263.0, 263.0],
        [-120.0, -120.0, 135.0, 135.0],
    ],
    [[-56.0, -56.0, 71.0, 71.0], [-24.0, -24.0, 39.0, 39.0]],
    [[-8.0, -8.0, 23.0, 23.0], [0.0, 0.0, 15.0, 15.0]],
];

const ARCFACE_DST: [(f32, f32); 5] = [
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
];

use crate::{
    AsVal, Model, Projection, QuantizationKind,
    bbox::BoundingBox,
    error::Error,
    model::{BindingInfo, IoBinding, IoBindingInfo, IoDefinition},
};

pub struct RetinafaceModel {
    threshold: f32,
    path: PathBuf,
    inputs: [IoDefinition; 1],
    inputs_info: RetinafaceInputsInfo,
    outputs_info: RetinafaceOutputsInfo,
    batch_size: u32,
    quatization: QuantizationKind,
    width: u32,
    height: u32,
}

impl RetinafaceModel {
    pub fn new(
        batch_size: u32,
        threshold: f32,
        quatization: QuantizationKind,
        path: impl Into<PathBuf>,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            threshold,
            path: path.into(),
            batch_size,
            inputs: [IoDefinition::new_static_name(
                INPUT_NAME,
                &[batch_size as _, 3, height as _, width as _],
                &[batch_size as _, 3, height as _, width as _],
                &[batch_size as _, 3, height as _, width as _],
            )],
            quatization,
            inputs_info: Default::default(),
            outputs_info: Default::default(),
            width,
            height,
        }
    }
}

pub struct RetinafaceInputs {
    input: HostBuffer<f32>,
}

impl IoBinding<f32> for RetinafaceInputs {
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

        self.input.num_elements()
    }

    #[inline]
    fn buffers(&self) -> impl Iterator<Item = (&str, &async_cuda::HostBuffer<f32>)> + Send {
        [(INPUT_NAME, &self.input)].into_iter()
    }

    #[inline]
    fn buffers_mut(
        &mut self,
    ) -> impl Iterator<Item = (&str, &mut async_cuda::HostBuffer<f32>)> + Send + Sync {
        [(INPUT_NAME, &mut self.input)].into_iter()
    }

    #[inline]
    fn get(&self, name: &str) -> &[f32] {
        assert_eq!(name, INPUT_NAME);

        self.input.as_slice()
    }

    #[inline]
    fn get_mut(&mut self, name: &str) -> &mut [f32] {
        assert_eq!(name, INPUT_NAME);

        self.input.as_mut_slice()
    }
}

#[derive(Clone, Default, Copy)]
pub struct RetinafaceInputsInfo {
    input: BindingInfo<4>,
}

impl IoBindingInfo for RetinafaceInputsInfo {
    #[inline]
    fn shape<const N: usize>(&self, name: &str) -> [usize; N] {
        assert_eq!(name, INPUT_NAME);
        self.input.into()
    }

    #[inline]
    fn shape_dyn(&self, name: &str) -> &[usize] {
        assert_eq!(name, INPUT_NAME);
        &self.input.shape
    }

    #[inline]
    fn size(&self, name: &str) -> usize {
        assert_eq!(name, INPUT_NAME);
        self.input.size
    }

    #[inline]
    fn set(&mut self, name: &str, shape: &[usize]) {
        assert_eq!(name, INPUT_NAME);
        self.input.set(shape)
    }
}

pub struct RetinafaceOutputs {
    bboxes: HostBuffer<f32>,
    probs: HostBuffer<f32>,
    landmarks: HostBuffer<f32>,
}

impl IoBinding<f32> for RetinafaceOutputs {
    #[inline]
    fn keys(&self) -> &[&str] {
        &[OUTPUT_NAME_BBOXES, OUTPUT_NAME_PROBS, OUTPUT_NAME_LANDMARKS]
    }

    #[inline]
    fn count(&self) -> usize {
        3
    }

    #[inline]
    fn size(&self, name: &str) -> usize {
        match name {
            OUTPUT_NAME_BBOXES => self.bboxes.num_elements(),
            OUTPUT_NAME_PROBS => self.probs.num_elements(),
            OUTPUT_NAME_LANDMARKS => self.landmarks.num_elements(),
            _ => unreachable!(),
        }
    }

    #[inline]
    fn buffers(&self) -> impl Iterator<Item = (&str, &async_cuda::HostBuffer<f32>)> + Send {
        [
            (OUTPUT_NAME_BBOXES, &self.bboxes),
            (OUTPUT_NAME_PROBS, &self.probs),
            (OUTPUT_NAME_LANDMARKS, &self.landmarks),
        ]
        .into_iter()
    }

    #[inline]
    fn buffers_mut(
        &mut self,
    ) -> impl Iterator<Item = (&str, &mut async_cuda::HostBuffer<f32>)> + Send + Sync {
        [
            (OUTPUT_NAME_BBOXES, &mut self.bboxes),
            (OUTPUT_NAME_PROBS, &mut self.probs),
            (OUTPUT_NAME_LANDMARKS, &mut self.landmarks),
        ]
        .into_iter()
    }

    #[inline]
    fn get(&self, name: &str) -> &[f32] {
        match name {
            OUTPUT_NAME_BBOXES => self.bboxes.as_slice(),
            OUTPUT_NAME_PROBS => self.probs.as_slice(),
            OUTPUT_NAME_LANDMARKS => self.landmarks.as_slice(),
            _ => unreachable!(),
        }
    }

    #[inline]
    fn get_mut(&mut self, name: &str) -> &mut [f32] {
        match name {
            OUTPUT_NAME_BBOXES => self.bboxes.as_mut_slice(),
            OUTPUT_NAME_PROBS => self.probs.as_mut_slice(),
            OUTPUT_NAME_LANDMARKS => self.landmarks.as_mut_slice(),
            _ => unreachable!(),
        }
    }
}

#[derive(Clone, Default, Copy)]
pub struct RetinafaceOutputsInfo {
    bboxes: BindingInfo<3>,
    probs: BindingInfo<3>,
    landmarks: BindingInfo<3>,
}

impl IoBindingInfo for RetinafaceOutputsInfo {
    #[inline]
    fn shape<const N: usize>(&self, name: &str) -> [usize; N] {
        match name {
            OUTPUT_NAME_BBOXES => self.bboxes.into(),
            OUTPUT_NAME_PROBS => self.probs.into(),
            OUTPUT_NAME_LANDMARKS => self.landmarks.into(),
            _ => unreachable!(),
        }
    }

    #[inline]
    fn shape_dyn(&self, name: &str) -> &[usize] {
        match name {
            OUTPUT_NAME_BBOXES => &self.bboxes.shape,
            OUTPUT_NAME_PROBS => &self.probs.shape,
            OUTPUT_NAME_LANDMARKS => &self.landmarks.shape,
            _ => unreachable!(),
        }
    }

    #[inline]
    fn size(&self, name: &str) -> usize {
        match name {
            OUTPUT_NAME_BBOXES => self.bboxes.size,
            OUTPUT_NAME_PROBS => self.probs.size,
            OUTPUT_NAME_LANDMARKS => self.landmarks.size,
            _ => unreachable!(),
        }
    }

    #[inline]
    fn set(&mut self, name: &str, shape: &[usize]) {
        match name {
            OUTPUT_NAME_BBOXES => self.bboxes.set(shape),
            OUTPUT_NAME_PROBS => self.probs.set(shape),
            OUTPUT_NAME_LANDMARKS => self.landmarks.set(shape),
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RetinaFaceDetection<I> {
    pub inner: Arc<I>,
    pub score: f32,
    pub bbox: BoundingBox,
    pub landmarks: [(f32, f32); 5],
}

impl<I> AsVal<Projection> for RetinaFaceDetection<I> {
    fn as_val(&self) -> Projection {
        todo!()
        // Projection {
        //     m: umeyama(&self.landmarks, &ARCFACE_DST),
        // }
    }
}

impl<I: AsRef<Rgb32FImage>> AsRef<Rgb32FImage> for RetinaFaceDetection<I> {
    fn as_ref(&self) -> &Rgb32FImage {
        self.inner.as_ref().as_ref()
    }
}

impl<I: AsRef<image::Rgb32FImage> + Send + Sync> Model<I> for RetinafaceModel {
    type TIn = f32;
    type TOut = f32;
    type Out = RetinaFaceDetection<I>;
    type Inputs = RetinafaceInputs;
    type Outputs = RetinafaceOutputs;
    type InputsInfo = RetinafaceInputsInfo;
    type OutputsInfo = RetinafaceOutputsInfo;

    #[inline]
    fn name(&self) -> &str {
        "retinaface"
    }

    #[inline]
    fn batch_size(&self) -> usize {
        self.batch_size as _
    }

    #[inline]
    fn inputs(&self) -> &[IoDefinition] {
        &self.inputs
    }

    #[inline]
    fn quantization(&self) -> QuantizationKind {
        self.quatization
    }

    #[inline]
    fn onnx_path(&self) -> &std::path::Path {
        &self.path
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
        RetinafaceInputs {
            input: HostBuffer::new(self.inputs_info.input.size).await,
        }
    }

    #[inline]
    async fn create_outputs(&self) -> Self::Outputs {
        RetinafaceOutputs {
            bboxes: HostBuffer::new(self.outputs_info.bboxes.size).await,
            probs: HostBuffer::new(self.outputs_info.probs.size).await,
            landmarks: HostBuffer::new(self.outputs_info.landmarks.size).await,
        }
    }

    fn pre_process(&self, input: &I, index: usize, inputs: &mut Self::Inputs) {
        let model_ratio = self.height as f32 / self.width as f32;
        let input_width = input.as_ref().width() as f32;
        let input_height = input.as_ref().height() as f32;
        let im_ratio = input_height / input_width;

        let (new_width, new_height) = if im_ratio > model_ratio {
            (self.height, (self.height as f32 / im_ratio) as u32)
        } else {
            (self.width, (self.width as f32 * im_ratio) as u32)
        };

        let mut input_1 = inputs.view_mut(INPUT_NAME, self.inputs_info.input.shape);
        let img = imageops::resize(input.as_ref(), new_width, new_height, FilterType::Triangle);

        for out_row in 0..self.width {
            for out_col in 0..self.height {
                if let Some(pixel) = img.get_pixel_checked(out_row as _, out_col as _) {
                    input_1[[index, 0, out_col as usize, out_row as usize]] =
                        (pixel.0[0] - 0.5) * 2.0;

                    input_1[[index, 1, out_col as usize, out_row as usize]] =
                        (pixel.0[1] - 0.5) * 2.0;

                    input_1[[index, 2, out_col as usize, out_row as usize]] =
                        (pixel.0[2] - 0.5) * 2.0;
                }
            }
        }

        // let img = imageops::resize(
        //     input.as_ref(),
        //     WIDTH as _,
        //     HEIGHT as _,
        //     FilterType::Triangle,
        // );

        // inputs
        //     .view_mut(INPUT_NAME, self.inputs_info.input.shape)
        //     .index_axis_mut(ndarray::Axis(0), index)
        //     .assign(
        //         &Array3::from_shape_vec([HEIGHT, WIDTH, 3], img.into_vec())
        //             .unwrap()
        //             .mapv(|x| x)
        //             .permuted_axes([2, 0, 1]),
        //     );
    }

    fn post_process(
        &self,
        input: I,
        index: usize,
        outputs: &Self::Outputs,
    ) -> impl futures::Stream<Item = Result<Self::Out, Error>> + Send {
        let _ = self.threshold;

        let orig_width = input.as_ref().width() as f32;
        let orig_height = input.as_ref().height() as f32;
        let _ratio = if orig_width / orig_height > 1. {
            orig_width / self.width as f32
        } else {
            orig_height / self.height as f32
        };

        // async_stream::stream! {
        let _input = Arc::new(input);
        // let mut faces = Vec::new();

        let confidences = outputs
            .view(OUTPUT_NAME_PROBS, self.outputs_info.probs.shape)
            .index_axis_move(Axis(0), index);

        let locations = outputs
            .view(OUTPUT_NAME_BBOXES, self.outputs_info.bboxes.shape)
            .index_axis_move(Axis(0), index);

        let landmarks = outputs
            .view(OUTPUT_NAME_LANDMARKS, self.outputs_info.landmarks.shape)
            .index_axis_move(Axis(0), index);

        println!("confidences: {:?}", confidences.shape());
        println!("locations: {:?}", locations.shape());
        println!("landmarks: {:?}", landmarks.shape());

        futures::stream::iter([])
    }
}

// pub fn infer(
//     landmarks: ArrayView3<f32>,
//     confidences: ArrayView3<f32>,
//     locations: ArrayView3<f32>,
//     image_size: [usize; 2],
// ) -> Result<Vec<(f32, [f32; 4], [(f32, f32); 5])>, Error> {
//     let confidence_threshold = 0.27016;
//     let nms_threshold = 0.2;
//     let variance = [0.1, 0.2];

//     let transformed_size = ndarray::Array::from_iter(image_size).into_owned();

//     let (prior_box, _) = prior_box(
//         ,
//         STRIDES,
//         false,
//         [image_size[0], image_size[1]],
//     );

//     let scale_landmarks = ndarray::concatenate(Axis(0), &[transformed_size.view(); 5])
//         .unwrap()
//         .mapv(|x| x as f32);

//     let scale_bboxes = ndarray::concatenate(Axis(0), &[transformed_size.view(); 2])
//         .unwrap()
//         .mapv(|x| x as f32);

//     let confidence_exp = confidences.map(|v| v.exp());
//     let confidences = &confidence_exp / confidence_exp.sum_axis(Axis(2)).insert_axis(Axis(2));

//     let mut boxes = decode(locations.slice(s![0, .., ..]), prior_box.view(), variance);

//     boxes = boxes * scale_bboxes;

//     let mut scores = confidences.slice(s![0, .., 1]).to_owned();

//     let mut landmarks = decode_landmark(
//         landmarks.slice(s![0, .., ..]).to_owned(),
//         prior_box.clone(),
//         variance,
//     );
//     landmarks = landmarks * scale_landmarks;

//     let valid_index = scores
//         .iter()
//         .enumerate()
//         .filter(|(_, val)| val > &&confidence_threshold)
//         .map(|(order, _)| order)
//         .collect::<Vec<_>>();

//     boxes = boxes.select(Axis(0), &valid_index);
//     landmarks = landmarks.select(Axis(0), &valid_index);
//     scores = scores.select(Axis(0), &valid_index);

//     let keep = nms(
//         &boxes,
//         &scores.mapv(|x| x as f64),
//         nms_threshold,
//         confidence_threshold as f64,
//     );

//     let mut faces = vec![];

//     for index in keep {
//         let bbox = boxes.slice(s![index, ..]);
//         let landmark = landmarks.slice(s![index, ..]);
//         faces.push((
//             scores[index],
//             [bbox[0], bbox[1], bbox[2], bbox[3]],
//             [
//                 (landmark[0], landmark[1]),
//                 (landmark[2], landmark[3]),
//                 (landmark[4], landmark[5]),
//                 (landmark[6], landmark[7]),
//                 (landmark[8], landmark[9]),
//             ],
//         ));
//     }

//     Ok(faces)
// }

pub fn nms<'a, N, BA, SA>(
    boxes: BA,
    scores: SA,
    iou_threshold: f64,
    score_threshold: f64,
) -> Vec<usize>
where
    N: Num + PartialEq + PartialOrd + ToPrimitive + Copy + PartialEq + 'a,
    BA: Into<ndarray::ArrayView2<'a, N>>,
    SA: Into<ndarray::ArrayView1<'a, f64>>,
{
    let boxes = boxes.into();
    let scores = scores.into();
    assert_eq!(boxes.nrows(), scores.len_of(Axis(0)));

    let order: Vec<usize> = {
        let mut indices: Vec<_> = if score_threshold > ZERO {
            // filter out boxes lower than score threshold
            scores
                .iter()
                .enumerate()
                .filter(|(_, score)| **score >= score_threshold)
                .map(|(idx, _)| idx)
                .collect()
        } else {
            (0..scores.len()).collect()
        };
        // sort box indices by scores
        indices.sort_unstable_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices
    };

    let mut keep: Vec<usize> = Vec::new();
    let mut suppress = vec![false; order.len()];

    for (i, &idx) in order.iter().enumerate() {
        if suppress[i] {
            continue;
        }
        keep.push(idx);
        let box1 = boxes.row(idx);
        let b1x = box1[0];
        let b1y = box1[1];
        let b1xx = box1[2];
        let b1yy = box1[3];
        let area1 = area(b1x, b1y, b1xx, b1yy);
        for j in (i + 1)..order.len() {
            if suppress[j] {
                continue;
            }
            let box2 = boxes.row(order[j]);
            let b2x = box2[0];
            let b2y = box2[1];
            let b2xx = box2[2];
            let b2yy = box2[3];

            // Intersection-over-union
            let x = max(b1x, b2x);
            let y = max(b1y, b2y);
            let xx = min(b1xx, b2xx);
            let yy = min(b1yy, b2yy);
            if x > xx || y > yy {
                // Boxes are not intersecting at all
                continue;
            };
            // Boxes are intersecting
            let intersection: N = area(x, y, xx, yy);
            let area2: N = area(b2x, b2y, b2xx, b2yy);
            let union: N = area1 + area2 - intersection;
            let iou: f64 = intersection.to_f64().unwrap() / union.to_f64().unwrap();
            if iou > iou_threshold {
                suppress[j] = true;
            }
        }
    }
    keep
}

pub const ONE: f64 = 1.0;
pub const ZERO: f64 = 0.0;

#[inline(always)]
pub fn area<N>(bx: N, by: N, bxx: N, byy: N) -> N
where
    N: Num + PartialEq + PartialOrd + ToPrimitive,
{
    (bxx - bx) * (byy - by)
}

#[inline(always)]
pub fn min<N>(a: N, b: N) -> N
where
    N: Num + PartialOrd,
{
    if a < b { a } else { b }
}

#[inline(always)]
pub fn max<N>(a: N, b: N) -> N
where
    N: Num + PartialOrd,
{
    if a > b { a } else { b }
}

/// Алгоритм `Кабша-Умеямы` - это метод нахождения оптимального перемещения, поворота
/// и масштабирования, который выравнивает два набора точек с минимальным среднеквадратичным отклонением (RMSD).
pub fn umeyama<const R: usize>(
    src: &[(f32, f32); R],
    dst: &[(f32, f32); R],
) -> nalgebra::Matrix3<f32> {
    let src_x_sum: f32 = src.iter().map(|v| v.0).sum();
    let src_x_mean = src_x_sum / (R as f32);

    let src_y_sum: f32 = src.iter().map(|v| v.1).sum();
    let src_y_mean = src_y_sum / (R as f32);

    let dst_x_sum: f32 = dst.iter().map(|v| v.0).sum();
    let dst_x_mean = dst_x_sum / (R as f32);

    let dst_y_sum: f32 = dst.iter().map(|v| v.1).sum();
    let dst_y_mean = dst_y_sum / (R as f32);

    let src_demean_s = nalgebra::ArrayStorage(src.map(|v| [v.0 - src_x_mean, v.1 - src_y_mean]));
    let dst_demean_s = nalgebra::ArrayStorage(dst.map(|v| [v.0 - dst_x_mean, v.1 - dst_y_mean]));

    let src_demean = nalgebra::Matrix::from_array_storage(src_demean_s);
    let dst_demean = nalgebra::Matrix::from_array_storage(dst_demean_s);

    let a = std::ops::Mul::mul(dst_demean, &src_demean.transpose()) / (R as f32);
    let svd = nalgebra::Matrix::svd(a, true, true);

    let determinant = a.determinant();

    let mut d = [1f32; 2];

    if determinant < 0.0f32 {
        d[2 - 1] = -1.0f32;
    }

    let mut t = nalgebra::Matrix2::<f32>::identity();
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
            let dg = nalgebra::Matrix2::<f32>::new(d[0], 0f32, 0f32, d[1]);

            let udg = u * dg;
            udg.mul_to(&v, &mut t);
            d[2 - 1] = s;
        }
    } else {
        let dg = nalgebra::Matrix2::<f32>::new(d[0], 0f32, 0f32, d[1]);
        let udg = u * dg;
        udg.mul_to(&v, &mut t);
    }

    let ddd = nalgebra::Matrix1x2::new(d[0], d[1]);
    let d_x_s = ddd * s;

    let var0 = src_demean.remove_row(0).variance();
    let var1 = src_demean.remove_row(1).variance();

    let varsum = var0 + var1;

    let scale = d_x_s.get((0, 0)).unwrap() / varsum;

    let dst_mean = nalgebra::Matrix2x1::<f32>::new(dst_x_mean, dst_y_mean);
    let src_mean = nalgebra::Matrix2x1::<f32>::new(src_x_mean, src_y_mean);
    let t_x_srcmean = t * src_mean;

    let xxx = scale * t_x_srcmean;
    let yyy = dst_mean - xxx;

    let m13 = *yyy.get(0).unwrap();
    let m23 = *yyy.get(1).unwrap();

    let m00x22 = t * scale;

    let m11 = m00x22.m11;
    let m21 = m00x22.m21;
    let m12 = m00x22.m12;
    let m22 = m00x22.m22;

    nalgebra::Matrix3::<f32>::new(m11, m12, m13, m21, m22, m23, 0f32, 0f32, 1f32)
}

fn generate_anchors(base_size: i32, ratios: &[f32], scales: &[f32]) -> Array2<f32> {
    let num_ratio = ratios.len();
    let num_scale = scales.len();
    let total = num_ratio * num_scale;

    // 4 rows (x1,y1,x2,y2) and total columns (one anchor per column)
    let mut anchors = Array2::<f32>::zeros((4, total));

    let cx = base_size as f32 * 0.5_f32;
    let cy = base_size as f32 * 0.5_f32;

    for (i, &ar) in ratios.iter().enumerate() {
        // compute r_w and r_h as in the C++ code
        let r_w = ((base_size as f32) / ar.sqrt()).round();
        let r_h = (r_w * ar).round();

        for (j, &scale) in scales.iter().enumerate() {
            let rs_w = r_w * scale;
            let rs_h = r_h * scale;

            let col = i * num_scale + j;
            anchors[[0, col]] = cx - rs_w * 0.5_f32;
            anchors[[1, col]] = cy - rs_h * 0.5_f32;
            anchors[[2, col]] = cx + rs_w * 0.5_f32;
            anchors[[3, col]] = cy + rs_h * 0.5_f32;
        }
    }

    anchors
}

#[derive(Clone, Debug)]
struct FaceObject {
    rect: BoundingBox,
    landmark: [Point2<f32>; 5],
    prob: f32,
}

fn process1(
    base_size: i32,
    score_blob: &Array3<f32>,
    bbox_blob: &Array3<f32>,
    landmark_blob: &Array3<f32>,
    prob_threshold: f32,
) -> Vec<FaceObject> {
    let mut faceproposals = Vec::<FaceObject>::new();

    for (idx, stride) in STRIDES.into_iter().enumerate() {
        let anchors = generate_anchors(base_size, &[1.0], &SCALES[idx]);

        generate_proposals(
            &anchors,
            stride,
            score_blob,
            bbox_blob,
            landmark_blob,
            prob_threshold,
            &mut faceproposals,
        );
    }

    faceproposals
}

/// anchors: shape (4, num_anchors) where each column is [x0,y0,x1,y1]
/// score_blob: shape (num_channels, h, w) - channel layout matches C++ usage
/// bbox_blob: shape (num_channels, h, w)
/// landmark_blob: shape (num_channels, h, w)
pub fn generate_proposals(
    anchors: &Array2<f32>,
    feat_stride: u32,
    score_blob: &Array3<f32>,
    bbox_blob: &Array3<f32>,
    landmark_blob: &Array3<f32>,
    prob_threshold: f32,
    faceobjects: &mut Vec<FaceObject>,
) {
    let w = score_blob.shape()[2];
    let h = score_blob.shape()[1];

    let num_anchors = anchors.shape()[1];

    for q in 0..num_anchors {
        // anchor column q
        let anchor_x0 = anchors[[0, q]];
        let anchor_y0 = anchors[[1, q]];

        let anchor_x1 = anchors[[2, q]];
        let anchor_y1 = anchors[[3, q]];

        // score channel for this anchor in C++ is channel(q + num_anchors)
        // bbox channels start at q*4, length 4
        // landmark channels start at q*10, length 10
        let score_channel_idx = q + num_anchors;
        let bbox_channel_start = q * 4;
        let landmark_channel_start = q * 10;

        // bounds check (silently skip if shapes don't match expected layout)
        if score_channel_idx >= score_blob.shape()[0] {
            continue;
        }

        if bbox_channel_start + 3 >= bbox_blob.shape()[0] {
            continue;
        }

        if landmark_channel_start + 9 >= landmark_blob.shape()[0] {
            continue;
        }

        let mut anchor_y = anchor_y0;
        let anchor_w = anchor_x1 - anchor_x0;
        let anchor_h = anchor_y1 - anchor_y0;

        for i in 0..h {
            let mut anchor_x = anchor_x0;

            for j in 0..w {
                let prob = score_blob[[score_channel_idx, i, j]];

                if prob >= prob_threshold {
                    // bbox deltas
                    let dx = bbox_blob[[bbox_channel_start + 0, i, j]];
                    let dy = bbox_blob[[bbox_channel_start + 1, i, j]];
                    let dw = bbox_blob[[bbox_channel_start + 2, i, j]];
                    let dh = bbox_blob[[bbox_channel_start + 3, i, j]];

                    let cx = anchor_x + anchor_w * 0.5_f32;
                    let cy = anchor_y + anchor_h * 0.5_f32;

                    let pb_cx = cx + anchor_w * dx;
                    let pb_cy = cy + anchor_h * dy;

                    let pb_w = anchor_w * (dw.exp());
                    let pb_h = anchor_h * (dh.exp());

                    let x0 = pb_cx - pb_w * 0.5_f32;
                    let y0 = pb_cy - pb_h * 0.5_f32;

                    let x1 = pb_cx + pb_w * 0.5_f32;
                    let y1 = pb_cy + pb_h * 0.5_f32;

                    // landmarks: 5 points, each has (x,y) stored in 10 channels
                    let lm0_x = landmark_blob[[landmark_channel_start + 0, i, j]];
                    let lm0_y = landmark_blob[[landmark_channel_start + 1, i, j]];
                    let lm1_x = landmark_blob[[landmark_channel_start + 2, i, j]];
                    let lm1_y = landmark_blob[[landmark_channel_start + 3, i, j]];
                    let lm2_x = landmark_blob[[landmark_channel_start + 4, i, j]];
                    let lm2_y = landmark_blob[[landmark_channel_start + 5, i, j]];
                    let lm3_x = landmark_blob[[landmark_channel_start + 6, i, j]];
                    let lm3_y = landmark_blob[[landmark_channel_start + 7, i, j]];
                    let lm4_x = landmark_blob[[landmark_channel_start + 8, i, j]];
                    let lm4_y = landmark_blob[[landmark_channel_start + 9, i, j]];

                    faceobjects.push(FaceObject {
                        rect: BoundingBox::new(x0, y0, x1, y1),
                        landmark: [
                            Point2::new(
                                cx + (anchor_w + 1.0) * lm0_x,
                                cy + (anchor_h + 1.0) * lm0_y,
                            ),
                            Point2::new(
                                cx + (anchor_w + 1.0) * lm1_x,
                                cy + (anchor_h + 1.0) * lm1_y,
                            ),
                            Point2::new(
                                cx + (anchor_w + 1.0) * lm2_x,
                                cy + (anchor_h + 1.0) * lm2_y,
                            ),
                            Point2::new(
                                cx + (anchor_w + 1.0) * lm3_x,
                                cy + (anchor_h + 1.0) * lm3_y,
                            ),
                            Point2::new(
                                cx + (anchor_w + 1.0) * lm4_x,
                                cy + (anchor_h + 1.0) * lm4_y,
                            ),
                        ],
                        prob,
                    });
                }

                anchor_x += feat_stride as f32;
            }

            anchor_y += feat_stride as f32;
        }
    }
}

//
// -----
//

pub struct RetinaModelConfig {
    name: String,
    min_sizes: [[u32; 2]; 3],
    steps: [u32; 3],
}

impl Default for RetinaModelConfig {
    fn default() -> Self {
        Self {
            name: Default::default(),
            min_sizes: PRIORS,
            steps: STRIDES,
        }
    }
}

fn prior_box(min_sizes: &[[u32; 2]], steps: &[u32], image_size: [u32; 2]) -> ndarray::Array2<f32> {
    let feature_maps = steps.iter().copied().map(|step| {
        [
            f32::ceil(image_size[0] as f32 / step as f32) as i32,
            f32::ceil(image_size[1] as f32 / step as f32) as i32,
        ]
    });

    let mut anchors: Vec<[f32; 4]> = vec![];
    for (k, f) in feature_maps.enumerate() {
        for (i, j) in Itertools::cartesian_product(0..f[0], 0..f[1]) {
            for min_size in min_sizes[k] {
                anchors.push([
                    (i as f32 + 0.5) * steps[k] as f32 / image_size[1] as f32,
                    (j as f32 + 0.5) * steps[k] as f32 / image_size[0] as f32,
                    min_size as f32 / image_size[0] as f32,
                    min_size as f32 / image_size[1] as f32,
                ]);
            }
        }
    }

    ndarray::arr2(&anchors).clamp(0.0, 1.0)
}

fn process(
    score_blob: &Array2<f32>,
    bbox_blob: &Array2<f32>,
    landmark_blob: &Array2<f32>,
    prob_threshold: f32,
    image_size: [u32; 2],
) -> Vec<FaceObject> {
    let min_sizes = PRIORS;
    let steps = STRIDES;

    let mut faceproposals = Vec::<FaceObject>::new();
    let anchors = prior_box(&min_sizes, &steps, image_size);

    decode(
        &mut faceproposals,
        score_blob,
        bbox_blob,
        landmark_blob,
        &anchors,
        [0.1, 0.2],
        prob_threshold,
    );

    faceproposals
}

fn decode(
    dst: &mut Vec<FaceObject>,
    probs: &Array2<f32>,
    bboxes: &Array2<f32>,
    landmarks: &Array2<f32>,
    priors: &Array2<f32>,
    variances: [f32; 2],
    threshold: f32,
) {
    let count = landmarks.shape()[0];
    dst.reserve(count);

    for i in 0..count {
        // confidence
        let prob = probs[[i, 0]];

        if prob < threshold {
            continue;
        }

        // priors
        let px = priors[[i, 0]];
        let py = priors[[i, 1]];
        let pw = priors[[i, 2]];
        let ph = priors[[i, 3]];
        let pw_var0 = pw * variances[0];
        let ph_var0 = ph * variances[0];

        // bboxes: each has (x,y,dw,dh) stored in 4 channels
        let dx = bboxes[[i, 0]];
        let dy = bboxes[[i, 1]];
        let dw = bboxes[[i, 2]];
        let dh = bboxes[[i, 3]];

        let cx = px + pw_var0 * dx;
        let cy = py + ph_var0 * dy;
        let w = pw * (variances[1] * dw).exp();
        let h = ph * (variances[1] * dh).exp();

        // landmarks: 5 points, each has (x,y) stored in 10 channels
        let lm0_x = landmarks[[i, 0]];
        let lm0_y = landmarks[[i, 1]];
        let lm1_x = landmarks[[i, 2]];
        let lm1_y = landmarks[[i, 3]];
        let lm2_x = landmarks[[i, 4]];
        let lm2_y = landmarks[[i, 5]];
        let lm3_x = landmarks[[i, 6]];
        let lm3_y = landmarks[[i, 7]];
        let lm4_x = landmarks[[i, 8]];
        let lm4_y = landmarks[[i, 9]];

        dst.push(FaceObject {
            prob,
            rect: BoundingBox::new2(cx - w * 0.5, cy - h * 0.5, w, h),
            landmark: [
                Point2::new(px + pw_var0 * lm0_x, py + ph_var0 * lm0_y),
                Point2::new(px + pw_var0 * lm1_x, py + ph_var0 * lm1_y),
                Point2::new(px + pw_var0 * lm2_x, py + ph_var0 * lm2_y),
                Point2::new(px + pw_var0 * lm3_x, py + ph_var0 * lm3_y),
                Point2::new(px + pw_var0 * lm4_x, py + ph_var0 * lm4_y),
            ],
        });
    }
}
