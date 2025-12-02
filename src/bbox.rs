use nalgebra::{Matrix3, Vector2};

use crate::{AsVal, Projection};

#[derive(Debug, Default, Clone, Copy)]
pub struct BoundingBox {
    pub(crate) x1: f32,
    pub(crate) y1: f32,
    pub(crate) x2: f32,
    pub(crate) y2: f32,
}

impl BoundingBox {
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self { x1, y1, x2, y2 }
    }

    #[inline]
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    #[inline]
    pub fn intersection(&self, other: &Self) -> Self {
        BoundingBox {
            x1: self.x1.max(other.x1),
            y1: self.y1.max(other.y1),
            x2: self.x2.min(other.x2),
            y2: self.y2.min(other.y2),
        }
    }

    #[inline]
    pub fn center_left(&self) -> f32 {
        self.left() + self.width() * 0.5
    }

    #[inline]
    pub fn center_top(&self) -> f32 {
        self.top() + self.height() * 0.5
    }

    #[inline]
    pub fn left(&self) -> f32 {
        self.x1
    }

    #[inline]
    pub fn top(&self) -> f32 {
        self.y1
    }

    #[inline]
    pub fn right(&self) -> f32 {
        self.x2
    }

    #[inline]
    pub fn bottom(&self) -> f32 {
        self.y2
    }

    #[inline]
    pub fn width(&self) -> f32 {
        self.x2 - self.x1
    }

    #[inline]
    pub fn height(&self) -> f32 {
        self.y2 - self.y1
    }

    #[inline]
    pub fn iou(&self, other: &Self) -> f32 {
        let int_area = self.intersection(other).area();

        int_area / (self.area() + other.area() - int_area + f32::EPSILON)
    }

    #[inline]
    pub fn scale(&mut self, x_scale: f32, y_scale: f32) {
        self.x1 *= x_scale;
        self.x2 *= x_scale;
        self.y1 *= y_scale;
        self.y2 *= y_scale;
    }

    #[inline]
    pub fn scaled(&self, x_scale: f32, y_scale: f32) -> Self {
        Self {
            x1: self.x1 * x_scale,
            x2: self.x2 * x_scale,
            y1: self.y1 * y_scale,
            y2: self.y2 * y_scale,
        }
    }

    #[inline]
    pub fn projection(&self) -> Projection {
        let scale = 112.0 / self.height();

        let t1 = Matrix3::new_translation(&Vector2::new(
            -(self.right() - self.width() / 2.0),
            -(self.bottom() - self.height() / 2.0),
        ));

        let s1 = Matrix3::new_scaling(scale);
        let t2 = Matrix3::new_translation(&Vector2::new(56.0, 56.0));

        Projection { m: t2 * s1 * t1 }
    }

    #[inline]
    pub fn map<F: Fn(f32) -> f32>(&self, map: F) -> Self {
        Self {
            x1: map(self.x1),
            y1: map(self.y1),
            x2: map(self.x2),
            y2: map(self.y2),
        }
    }
}

impl AsVal<Projection> for BoundingBox {
    fn as_val(&self) -> Projection {
        self.projection()
    }
}
