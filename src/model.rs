use std::{
    borrow::Cow,
    hash::{DefaultHasher, Hash, Hasher},
    path::Path,
    pin::Pin,
};

use async_cuda::HostBuffer;
use futures::{Stream, TryStreamExt};

use crate::{AsVal, QuantizationKind, error::Error};

pub mod diffiqa;
// pub mod insightface;
pub mod retinaface;
pub mod yolo;

#[derive(Debug, Clone, Copy)]
struct BindingInfo<const N: usize> {
    pub size: usize,
    pub shape: [usize; N],
}

impl<const N: usize> BindingInfo<N> {
    #[inline]
    pub fn set(&mut self, shape: &[usize]) {
        self.shape.copy_from_slice(shape);
        self.update_size();
    }

    #[inline]
    fn update_size(&mut self) {
        self.size = self.shape.iter().copied().product();
    }
}

impl<const N: usize> Default for BindingInfo<N> {
    fn default() -> Self {
        Self {
            size: Default::default(),
            shape: [0; N],
        }
    }
}

impl<const A: usize, const B: usize> From<BindingInfo<B>> for [usize; A] {
    fn from(value: BindingInfo<B>) -> Self {
        let mut x = [0; A];
        x.copy_from_slice(&value.shape);
        x
    }
}

#[derive(Debug, Clone, Hash)]
pub struct IoDefinition {
    pub name: Cow<'static, str>,
    pub shape_min: Cow<'static, [i32]>,
    pub shape_opt: Cow<'static, [i32]>,
    pub shape_max: Cow<'static, [i32]>,
}

impl IoDefinition {
    pub const fn new_static(
        name: &'static str,
        shape_min: &'static [i32],
        shape_opt: &'static [i32],
        shape_max: &'static [i32],
    ) -> Self {
        Self {
            name: Cow::Borrowed(name),
            shape_min: Cow::Borrowed(shape_min),
            shape_opt: Cow::Borrowed(shape_opt),
            shape_max: Cow::Borrowed(shape_max),
        }
    }

    pub fn new_static_name(
        name: &'static str,
        shape_min: &[i32],
        shape_opt: &[i32],
        shape_max: &[i32],
    ) -> Self {
        Self {
            name: Cow::Borrowed(name),
            shape_min: Cow::Owned(shape_min.to_vec()),
            shape_opt: Cow::Owned(shape_opt.to_vec()),
            shape_max: Cow::Owned(shape_max.to_vec()),
        }
    }

    pub fn new(name: &str, shape_min: &[i32], shape_opt: &[i32], shape_max: &[i32]) -> Self {
        Self {
            name: Cow::Owned(name.to_string()),
            shape_min: Cow::Owned(shape_min.to_vec()),
            shape_opt: Cow::Owned(shape_opt.to_vec()),
            shape_max: Cow::Owned(shape_max.to_vec()),
        }
    }
}

pub trait IoBindingInfo: Send + 'static {
    fn shape<const N: usize>(&self, name: &str) -> [usize; N]
    where
        Self: Sized;
    fn shape_dyn(&self, name: &str) -> &[usize];
    fn size(&self, name: &str) -> usize;
    fn set(&mut self, name: &str, shape: &[usize]);
}

pub trait IoBinding<T>: Send + Sync + 'static
where
    T: Copy + 'static,
{
    fn keys(&self) -> &[&str];
    fn count(&self) -> usize;
    fn size(&self, name: &str) -> usize;
    fn buffers(&self) -> impl Iterator<Item = (&str, &HostBuffer<T>)> + Send;
    fn buffers_mut(&mut self) -> impl Iterator<Item = (&str, &mut HostBuffer<T>)> + Send + Sync;

    fn get(&self, name: &str) -> &[T];
    fn get_mut(&mut self, name: &str) -> &mut [T];

    #[inline]
    fn view<D: ndarray::Dimension>(
        &self,
        name: &str,
        shape: impl Into<ndarray::StrideShape<D>>,
    ) -> ndarray::ArrayView<'_, T, D> {
        ndarray::ArrayView::from_shape(shape, self.get(name)).unwrap()
    }

    #[inline]
    fn view_mut<D: ndarray::Dimension>(
        &mut self,
        name: &str,
        shape: impl Into<ndarray::StrideShape<D>>,
    ) -> ndarray::ArrayViewMut<'_, T, D> {
        ndarray::ArrayViewMut::from_shape(shape, self.get_mut(name)).unwrap()
    }
}

pub trait Model<In>: Send + Sync {
    type TIn: Copy + 'static;
    type TOut: Copy + 'static;
    type Out: Send;
    type Inputs: IoBinding<Self::TIn>;
    type Outputs: IoBinding<Self::TOut>;
    type InputsInfo: IoBindingInfo;
    type OutputsInfo: IoBindingInfo;

    fn name(&self) -> &str;
    fn inputs(&self) -> &[IoDefinition];
    fn onnx_path(&self) -> &Path;
    fn batch_size(&self) -> usize {
        1
    }

    fn quantization(&self) -> QuantizationKind {
        QuantizationKind::Float32
    }

    fn fingerprint(&self) -> u64 {
        let mut state = DefaultHasher::new();
        self.quantization().hash(&mut state);
        self.batch_size().hash(&mut state);
        self.inputs().hash(&mut state);
        self.onnx_path().hash(&mut state);
        state.finish()
    }

    fn inputs_info(&mut self) -> &mut dyn IoBindingInfo;
    fn outputs_info(&mut self) -> &mut dyn IoBindingInfo;

    fn create_inputs(&self) -> impl Future<Output = Self::Inputs>;
    fn create_outputs(&self) -> impl Future<Output = Self::Outputs>;

    fn pre_process(&self, input: &In, index: usize, dst: &mut Self::Inputs);
    fn post_process(
        &self,
        input: In,
        index: usize,
        outputs: &Self::Outputs,
    ) -> impl Stream<Item = Result<Self::Out, Error>> + Send;
}

pub enum Either<L, R> {
    Left(L),
    Right(R),
}

impl<U, L: AsRef<U>, R: AsRef<U>> AsRef<U> for Either<L, R> {
    fn as_ref(&self) -> &U {
        match self {
            Either::Left(l) => l.as_ref(),
            Either::Right(r) => r.as_ref(),
        }
    }
}

impl<U, L: AsMut<U>, R: AsMut<U>> AsMut<U> for Either<L, R> {
    fn as_mut(&mut self) -> &mut U {
        match self {
            Either::Left(l) => l.as_mut(),
            Either::Right(r) => r.as_mut(),
        }
    }
}

impl<U, L: AsVal<U>, R: AsVal<U>> AsVal<U> for Either<L, R> {
    fn as_val(&self) -> U {
        match self {
            Either::Left(l) => l.as_val(),
            Either::Right(r) => r.as_val(),
        }
    }
}

impl<L: Iterator, R: Iterator<Item = L::Item>> Iterator for Either<L, R> {
    type Item = L::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Either::Left(i) => i.next(),
            Either::Right(i) => i.next(),
        }
    }
}

impl<L: Stream, R: Stream<Item = L::Item>> Stream for Either<L, R> {
    type Item = L::Item;

    #[inline]
    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        match unsafe { self.get_unchecked_mut() } {
            Either::Left(s) => unsafe { Pin::new_unchecked(s).poll_next(cx) },
            Either::Right(s) => unsafe { Pin::new_unchecked(s).poll_next(cx) },
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Either::Left(s) => s.size_hint(),
            Either::Right(s) => s.size_hint(),
        }
    }
}

impl<L: IoBindingInfo, R: IoBindingInfo> IoBindingInfo for Either<L, R> {
    #[inline]
    fn shape<const N: usize>(&self, name: &str) -> [usize; N] {
        match self {
            Either::Left(io) => io.shape(name),
            Either::Right(io) => io.shape(name),
        }
    }

    #[inline]
    fn shape_dyn(&self, name: &str) -> &[usize] {
        match self {
            Either::Left(io) => io.shape_dyn(name),
            Either::Right(io) => io.shape_dyn(name),
        }
    }

    #[inline]
    fn size(&self, name: &str) -> usize {
        match self {
            Either::Left(io) => io.size(name),
            Either::Right(io) => io.size(name),
        }
    }

    #[inline]
    fn set(&mut self, name: &str, shape: &[usize]) {
        match self {
            Either::Left(io) => io.set(name, shape),
            Either::Right(io) => io.set(name, shape),
        }
    }
}

impl<T: Send + Copy + 'static, L: IoBinding<T>, R: IoBinding<T>> IoBinding<T> for Either<L, R> {
    #[inline]
    fn keys(&self) -> &[&str] {
        match self {
            Either::Left(io) => io.keys(),
            Either::Right(io) => io.keys(),
        }
    }

    #[inline]
    fn count(&self) -> usize {
        match self {
            Either::Left(io) => io.count(),
            Either::Right(io) => io.count(),
        }
    }

    #[inline]
    fn size(&self, name: &str) -> usize {
        match self {
            Either::Left(io) => io.size(name),
            Either::Right(io) => io.size(name),
        }
    }

    #[inline]
    fn buffers(&self) -> impl Iterator<Item = (&str, &HostBuffer<T>)> + Send {
        match self {
            Either::Left(io) => Either::Left(io.buffers()),
            Either::Right(io) => Either::Right(io.buffers()),
        }
    }

    #[inline]
    fn buffers_mut(&mut self) -> impl Iterator<Item = (&str, &mut HostBuffer<T>)> + Send + Sync {
        match self {
            Either::Left(io) => Either::Left(io.buffers_mut()),
            Either::Right(io) => Either::Right(io.buffers_mut()),
        }
    }

    #[inline]
    fn get(&self, name: &str) -> &[T] {
        match self {
            Either::Left(io) => io.get(name),
            Either::Right(io) => io.get(name),
        }
    }

    #[inline]
    fn get_mut(&mut self, name: &str) -> &mut [T] {
        match self {
            Either::Left(io) => io.get_mut(name),
            Either::Right(io) => io.get_mut(name),
        }
    }
}

impl<L, R> Either<L, R> {
    #[inline]
    fn left_ref(&self) -> &L {
        match self {
            Either::Left(io) => io,
            Either::Right(_) => unreachable!(),
        }
    }

    #[inline]
    fn left_mut(&mut self) -> &mut L {
        match self {
            Either::Left(io) => io,
            Either::Right(_) => unreachable!(),
        }
    }

    #[inline]
    fn right_ref(&self) -> &R {
        match self {
            Either::Left(_) => unreachable!(),
            Either::Right(io) => io,
        }
    }

    #[inline]
    fn right_mut(&mut self) -> &mut R {
        match self {
            Either::Left(_) => unreachable!(),
            Either::Right(io) => io,
        }
    }
}

impl<I, L, R> Model<I> for Either<L, R>
where
    L: Model<I>,
    R: Model<I, TIn = L::TIn, TOut = L::TOut>,
    L::TIn: Send + Copy + 'static,
    L::TOut: Send + Copy + 'static,
    L::Inputs: IoBinding<L::TIn>,
    R::Inputs: IoBinding<L::TIn>,
    L::Outputs: IoBinding<L::TOut>,
    R::Outputs: IoBinding<L::TOut>,
    L::InputsInfo: IoBindingInfo,
    R::InputsInfo: IoBindingInfo,
    L::OutputsInfo: IoBindingInfo,
    R::OutputsInfo: IoBindingInfo,
{
    type TIn = L::TIn;
    type TOut = L::TOut;
    type Out = Either<L::Out, R::Out>;
    type Inputs = Either<L::Inputs, R::Inputs>;
    type Outputs = Either<L::Outputs, R::Outputs>;
    type InputsInfo = Either<L::InputsInfo, R::InputsInfo>;
    type OutputsInfo = Either<L::OutputsInfo, R::OutputsInfo>;

    #[inline]
    fn name(&self) -> &str {
        match self {
            Either::Left(m) => m.name(),
            Either::Right(m) => m.name(),
        }
    }

    #[inline]
    fn inputs(&self) -> &[IoDefinition] {
        match self {
            Either::Left(m) => m.inputs(),
            Either::Right(m) => m.inputs(),
        }
    }

    #[inline]
    fn batch_size(&self) -> usize {
        match self {
            Either::Left(m) => m.batch_size(),
            Either::Right(m) => m.batch_size(),
        }
    }

    #[inline]
    fn quantization(&self) -> QuantizationKind {
        match self {
            Either::Left(m) => m.quantization(),
            Either::Right(m) => m.quantization(),
        }
    }

    #[inline]
    fn fingerprint(&self) -> u64 {
        match self {
            Either::Left(m) => m.fingerprint(),
            Either::Right(m) => m.fingerprint(),
        }
    }

    #[inline]
    fn onnx_path(&self) -> &Path {
        match self {
            Either::Left(m) => m.onnx_path(),
            Either::Right(m) => m.onnx_path(),
        }
    }

    #[inline]
    fn inputs_info(&mut self) -> &mut dyn IoBindingInfo {
        match self {
            Either::Left(m) => m.inputs_info(),
            Either::Right(m) => m.inputs_info(),
        }
    }

    #[inline]
    fn outputs_info(&mut self) -> &mut dyn IoBindingInfo {
        match self {
            Either::Left(m) => m.outputs_info(),
            Either::Right(m) => m.outputs_info(),
        }
    }

    #[inline]
    async fn create_inputs(&self) -> Self::Inputs {
        match self {
            Either::Left(m) => Either::Left(m.create_inputs().await),
            Either::Right(m) => Either::Right(m.create_inputs().await),
        }
    }

    #[inline]
    async fn create_outputs(&self) -> Self::Outputs {
        match self {
            Either::Left(m) => Either::Left(m.create_outputs().await),
            Either::Right(m) => Either::Right(m.create_outputs().await),
        }
    }

    #[inline]
    fn pre_process(&self, input: &I, index: usize, dst: &mut Self::Inputs) {
        match self {
            Either::Left(m) => m.pre_process(input, index, dst.left_mut()),
            Either::Right(m) => m.pre_process(input, index, dst.right_mut()),
        }
    }

    #[inline]
    fn post_process(
        &self,
        input: I,
        index: usize,
        outputs: &Self::Outputs,
    ) -> impl Stream<Item = Result<Self::Out, Error>> + Send {
        match self {
            Either::Left(m) => Either::Left(
                m.post_process(input, index, outputs.left_ref())
                    .map_ok(Either::Left),
            ),
            Either::Right(m) => Either::Right(
                m.post_process(input, index, outputs.right_ref())
                    .map_ok(Either::Right),
            ),
        }
    }
}
