use std::time::SystemTime;

pub fn _time_it<T>(func: impl FnOnce() -> T) {
    let s = SystemTime::now();
    func();
    let e = SystemTime::now();
    let duration = e.duration_since(s).unwrap();
    println!("Time Elapsed: {} seconds", duration.as_secs_f64())
}
