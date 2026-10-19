//! The pen's barrel (side) button, from the per-event pen flags a platform reports: whether it is
//! held, and a click with it (pressed and released within [`CLICK_MS`]), which Windows Ink and
//! tablet drivers treat as a right-click. Whether the tip touches doesn't matter: Wacom's driver
//! with the side switch set to Right Click reports the pen in contact for the emulated click, and
//! the window gets no press of its own for it. Pure, so it is tested on every platform;
//! [`crate::windows`] feeds it.

/// Longest press of the button that still counts as a click, in milliseconds. A longer hold is
/// the button held for something else (a right-drag), not a click.
pub const CLICK_MS: u32 = 600;

/// What the barrel button is doing, reported whenever it changes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Buttons {
    /// The barrel button is held.
    pub barrel: bool,
    /// It was just clicked: pressed and released within [`CLICK_MS`].
    pub click: bool,
}

/// Tracks the barrel button across pen events.
#[derive(Clone, Copy, Debug, Default)]
pub struct Tracker {
    /// When the button went down (the event's time in milliseconds), while it is held.
    down_at: Option<u32>,
}

impl Tracker {
    /// Nothing held (usable in a `static`).
    pub const DEFAULT: Tracker = Tracker { down_at: None };

    /// One pen event: the button's state at `time_ms` (a millisecond clock that may wrap, like
    /// Windows' message time). `Some` when the state changed.
    pub fn update(&mut self, barrel: bool, time_ms: u32) -> Option<Buttons> {
        match (self.down_at, barrel) {
            (None, true) => {
                self.down_at = Some(time_ms);
                Some(Buttons { barrel: true, click: false })
            }
            (Some(at), false) => {
                self.down_at = None;
                Some(Buttons { barrel: false, click: time_ms.wrapping_sub(at) <= CLICK_MS })
            }
            _ => None,
        }
    }

    /// The pen left the window or its range: the button can't still be held (no click).
    pub fn leave(&mut self) -> Option<Buttons> {
        self.down_at.take().map(|_| Buttons::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_quick_press_and_release_is_a_click() {
        let mut t = Tracker::default();
        assert_eq!(t.update(false, 0), None);
        assert_eq!(t.update(true, 1000), Some(Buttons { barrel: true, click: false }));
        assert_eq!(t.update(true, 1050), None, "no change, nothing to report");
        assert_eq!(t.update(false, 1120), Some(Buttons { barrel: false, click: true }));
    }

    #[test]
    fn a_long_hold_is_no_click() {
        let mut t = Tracker::default();
        t.update(true, 0);
        assert_eq!(t.update(false, CLICK_MS + 1), Some(Buttons { barrel: false, click: false }));
    }

    #[test]
    fn the_clock_may_wrap() {
        let mut t = Tracker::default();
        t.update(true, u32::MAX - 10);
        assert_eq!(t.update(false, 50), Some(Buttons { barrel: false, click: true }));
    }

    #[test]
    fn leaving_releases_the_button_without_a_click() {
        let mut t = Tracker::default();
        t.update(true, 0);
        assert_eq!(t.leave(), Some(Buttons::default()));
        assert_eq!(t.leave(), None);
        assert_eq!(t.update(false, 10), None);
    }
}
