use std::time::Duration;

/// Interpolation curves used by [`Tween`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Easing {
    /// Starts slowly and accelerates toward the target.
    InQuad,
    /// Starts quickly and slows toward the target using a quadratic curve.
    OutQuad,
    /// Starts quickly and slows toward the target.
    OutCubic,
}

impl Easing {
    /// Applies this easing curve to normalized progress in the range `0.0..=1.0`.
    ///
    /// NaN progress is treated as complete progress, while values outside the
    /// range are clamped to the nearest endpoint.
    pub fn apply(self, progress: f64) -> f64 {
        let progress = if progress.is_nan() {
            1.0
        } else {
            progress.clamp(0.0, 1.0)
        };

        match self {
            Self::InQuad => progress * progress,
            Self::OutQuad => 1.0 - (1.0 - progress).powi(2),
            Self::OutCubic => 1.0 - (1.0 - progress).powi(3),
        }
    }
}

/// A time-based interpolation between two scalar values.
#[derive(Debug, Clone, Copy)]
pub struct Tween {
    from: f64,
    to: f64,
    start: Duration,
    duration: Duration,
    easing: Easing,
}

impl Default for Tween {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl Tween {
    /// Creates an inactive tween whose current and target values are `value`.
    pub fn new(value: f64) -> Self {
        Self {
            from: value,
            to: value,
            start: Duration::ZERO,
            duration: Duration::ZERO,
            easing: Easing::OutQuad,
        }
    }

    /// Returns the interpolated value at the specified time.
    pub fn value_at(&self, elapsed: Duration) -> f64 {
        if self.duration.is_zero() || elapsed >= self.start.saturating_add(self.duration) {
            return self.to;
        }
        let progress =
            elapsed.saturating_sub(self.start).as_secs_f64() / self.duration.as_secs_f64();
        self.from + (self.to - self.from) * self.easing.apply(progress)
    }

    /// Starts an interpolation toward `to` from the value at `elapsed`.
    ///
    /// The current value is captured before applying the new duration and
    /// easing, allowing a new animation to begin from the current value.
    pub fn animate_to(&mut self, to: f64, elapsed: Duration, duration: Duration, easing: Easing) {
        let from = self.value_at(elapsed);
        *self = Self {
            from,
            to,
            start: elapsed,
            duration,
            easing,
        };
    }

    /// Returns the target value of this tween.
    pub fn target_value(&self) -> f64 {
        self.to
    }

    /// Returns whether `elapsed` is before the tween's end time.
    pub fn is_animating(&self, elapsed: Duration) -> bool {
        elapsed < self.start.saturating_add(self.duration)
    }
}
