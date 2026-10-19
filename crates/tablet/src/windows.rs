//! The pen's barrel button on Windows. winit 0.30 turns `WM_POINTER` pen input into touch events
//! and drops `POINTER_PEN_INFO::penFlags`, so with Windows Ink a press of the side button reaches
//! the app as nothing at all, and a tap with it held as an ordinary left click (which paints). This
//! module reads the flags itself from winit's message hook (`EventLoopBuilderExtWindows::
//! with_msg_hook`), which sees each queued message before winit handles it, so the button state is
//! current when the matching pointer event reaches the UI. It never consumes a message.
//!
//! Besides `crate::macos`, the only module that allows `unsafe`: one call to `GetPointerPenInfo`
//! (user32), with a `SAFETY:` comment. The button logic is the pure [`crate::pen_buttons`].

#![allow(unsafe_code)]

use std::ffi::c_void;
use std::sync::{Mutex, PoisonError};

use crate::pen_buttons::{Buttons, Tracker};

const WM_POINTERUPDATE: u32 = 0x0245;
const WM_POINTERDOWN: u32 = 0x0246;
const WM_POINTERUP: u32 = 0x0247;
const WM_POINTERLEAVE: u32 = 0x024A;
const WM_POINTERCAPTURECHANGED: u32 = 0x024C;
const PEN_FLAG_BARREL: u32 = 0x0000_0001;
const POINTER_FLAG_INCONTACT: u32 = 0x0000_0004;
const POINTER_FLAG_SECONDBUTTON: u32 = 0x0000_0020;
/// `POINTER_BUTTON_CHANGE_TYPE::POINTER_CHANGE_SECONDBUTTON_DOWN`.
const POINTER_CHANGE_SECONDBUTTON_DOWN: i32 = 3;

#[repr(C)]
struct Point {
    x: i32,
    y: i32,
}

/// `MSG` (winuser.h).
#[repr(C)]
struct Msg {
    hwnd: *mut c_void,
    message: u32,
    wparam: usize,
    lparam: isize,
    time: u32,
    pt: Point,
}

/// `POINTER_INFO` (winuser.h).
#[repr(C)]
struct PointerInfo {
    pointer_type: u32,
    pointer_id: u32,
    frame_id: u32,
    pointer_flags: u32,
    source_device: *mut c_void,
    hwnd_target: *mut c_void,
    pixel: Point,
    himetric: Point,
    pixel_raw: Point,
    himetric_raw: Point,
    time: u32,
    history_count: u32,
    input_data: i32,
    key_states: u32,
    performance_count: u64,
    button_change_type: i32,
}

/// `POINTER_PEN_INFO` (winuser.h).
#[repr(C)]
struct PointerPenInfo {
    pointer_info: PointerInfo,
    pen_flags: u32,
    pen_mask: u32,
    pressure: u32,
    rotation: u32,
    tilt_x: i32,
    tilt_y: i32,
}

#[link(name = "user32")]
unsafe extern "system" {
    fn GetPointerPenInfo(pointer_id: u32, pen_info: *mut PointerPenInfo) -> i32;
}

type Sink = Box<dyn Fn(Buttons) + Send>;

struct State {
    tracker: Tracker,
    sink: Option<Sink>,
}

static STATE: Mutex<State> = Mutex::new(State { tracker: Tracker::DEFAULT, sink: None });

/// Where button changes go once the app exists (the hook runs from the first message on).
pub fn set_sink(sink: impl Fn(Buttons) + Send + 'static) {
    STATE.lock().unwrap_or_else(PoisonError::into_inner).sink = Some(Box::new(sink));
}

/// Whether the pen's side button is down: the barrel flag, or the secondary ("right") button a
/// tablet driver reports instead when the side switch is set to Right Click (Wacom with Windows
/// Ink sends the click as a pointer event with the second button, not always with the flag).
fn side_button(info: &PointerPenInfo) -> bool {
    info.pen_flags & PEN_FLAG_BARREL != 0
        || info.pointer_info.pointer_flags & POINTER_FLAG_SECONDBUTTON != 0
        || info.pointer_info.button_change_type == POINTER_CHANGE_SECONDBUTTON_DOWN
}

/// Whether the pen's side button is down for pointer `id`, or `None` when it isn't a pen (a
/// mouse, a finger).
fn pen_flags(id: u32) -> Option<bool> {
    // SAFETY: `info` is a properly aligned, writable `POINTER_PEN_INFO` that lives across the
    // call, and `GetPointerPenInfo` writes at most that struct. It fails (returns 0) for an id
    // that isn't a pen or is no longer current, which is reported as `None`.
    let mut info: PointerPenInfo = unsafe { std::mem::zeroed() };
    let ok = unsafe { GetPointerPenInfo(id, &mut info) } != 0;
    if !ok {
        return None;
    }
    let barrel = side_button(&info);
    // What the driver reports, for a log to diagnose a side button that does nothing.
    log::debug!(
        "pen: penFlags {:#x}, pointerFlags {:#x} (in contact: {}), buttonChange {}",
        info.pen_flags,
        info.pointer_info.pointer_flags,
        info.pointer_info.pointer_flags & POINTER_FLAG_INCONTACT != 0,
        info.pointer_info.button_change_type
    );
    Some(barrel)
}

/// winit's message hook: reads the barrel button from pen pointer messages. Always returns
/// `false`, so winit handles every message as usual.
pub fn msg_hook(msg: *const c_void) -> bool {
    if msg.is_null() {
        return false;
    }
    // A panic must not unwind into winit's message loop.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: winit passes a pointer to the `MSG` it just took off the queue, valid for the
        // duration of the hook call; only its plain fields are read.
        let (message, wparam, time) = unsafe {
            let m = &*(msg as *const Msg);
            (m.message, m.wparam, m.time)
        };
        let change = match message {
            WM_POINTERUPDATE | WM_POINTERDOWN | WM_POINTERUP => {
                let Some(barrel) = pen_flags((wparam & 0xFFFF) as u32) else { return };
                STATE.lock().unwrap_or_else(PoisonError::into_inner).tracker.update(barrel, time)
            }
            WM_POINTERLEAVE | WM_POINTERCAPTURECHANGED => STATE.lock().unwrap_or_else(PoisonError::into_inner).tracker.leave(),
            _ => None,
        };
        if let Some(b) = change {
            log::info!("pen side button {}{}", if b.barrel { "down" } else { "up" }, if b.click { " (click: opens the brush picker)" } else { "" });
            let st = STATE.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some(sink) = &st.sink {
                sink(b);
            }
        }
    }));
    false
}
