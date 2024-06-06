use std::f64::consts::PI;

use arrayvec::ArrayVec;
// use pyo3::prelude::*;
use ::rayon::prelude::*;

#[non_exhaustive]
struct Constants;

impl Constants {
    pub const SMALL: f64 = 1e-10;
    pub const TINY: f64 = 1e-100;
    pub const MINISCULE: f64 = 1e-200;
}

#[derive(Debug, Clone)]
struct Centroid {
    value: f64,
    weight: f64,
}

impl Centroid {

    fn reset(&mut self) {
        self.value = 0.0;
        self.weight = 0.0;
    }
}


impl PartialEq for Centroid{

    fn eq(&self, other: &Self) -> bool {
        
        self.value.eq(&other.value)
    }

}

impl Eq for Centroid {}

impl PartialOrd for Centroid {

    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        
        self.value.partial_cmp(&other.value)
    }
}

impl Ord for Centroid {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.value.partial_cmp(&other.value) {
            Some(ordering) => ordering,
            None => {
                if self.value.is_nan() {
                    if other.value.is_nan() {
                        std::cmp::Ordering::Equal
                    } else {
                        std::cmp::Ordering::Greater
                    }
                } else {
                    std::cmp::Ordering::Less
                }
            }
        }
    }
}


struct Container {
    centroids: ArrayVec<Centroid, 10000>,
}

impl Container {

    fn new() -> Container {

        let centroids: ArrayVec<Centroid, 10000> = ArrayVec::<Centroid, 10000>::new();

        Container {centroids}
    }

    // fn is_full(&self) -> bool {
    //     self.centroids.is_full()
    // }

    fn weigh(&self) -> f64 {

        let mut out: f64 = 0.0;

        for centroid in &self.centroids {
            out += centroid.weight
        }
    
        return out
    }

    fn size(&self) -> usize {
        self.centroids.len()
    }

    fn push(&mut self, centroid: Centroid) {
        self.centroids.push(centroid);
    }

}

struct Digest {

    delta: f64,
    mean: f64,
    variance: f64,
    weight: f64,
    processed: bool,
    
    merged: Container,
    unmerged: Container,
}

impl Digest {

    fn new(delta: f64) -> Digest {

        assert!(delta > 0.0);

        // let capacity: usize = (6.0 * delta + 10.0) as usize;

        Digest {
            delta,
            mean: 0.0,
            variance: 1.0,
            weight: 1.0,
            processed: true,
            merged: Container::new(),
            unmerged: Container::new(),
        }

    }

    fn _should_merge(&self) -> bool {

        if self.processed {
            return false
        }

        (self.merged.centroids.remaining_capacity() - self.merged.centroids.capacity() < 10)
        | (self.unmerged.centroids.remaining_capacity() - self.unmerged.centroids.capacity() < 10)

    }

    fn push(&mut self, value: f64, weight: f64) {

        let delta: f64 = value - self.mean;

        let ratio: f64 = weight / (weight + self.merged.weigh());

        self.unmerged.push(Centroid{value, weight});

        self.mean += delta * ratio;

        self.variance = (1. - ratio) * (self.variance + ratio * delta * delta);

        self.processed = false;

    }

    fn append(&mut self, values: Vec<f64>, weight: f64) {

        for value in values {
            self.push(value, weight)
        }
    }

    fn merge(&mut self) {

        if self.unmerged.size() == 0 {
            return
        }


        for centroid in &self.unmerged.centroids {

            self.merged.centroids.push(centroid.clone());
        }

        self.merged.centroids.sort();

        let weight: f64 = self.merged.weigh();

        let denominator: f64 = 2. * PI * weight * (weight + Constants::SMALL).ln();

        let normalizer: f64 = self.delta / denominator;

        let mut other: usize = 0;
        let mut cum: f64 = 0.;

        for index in 1..self.merged.size() {

            let proposition: f64 = self.merged.centroids[index].weight + self.merged.centroids[other].weight;

            let is_too_small: bool = (self.merged.centroids[index].weight < Constants::MINISCULE) | (self.merged.centroids[other].weight < Constants::MINISCULE);
            let is_too_close: bool = ((self.merged.centroids[index].weight - self.merged.centroids[other].weight).abs() < Constants::TINY) | ((self.merged.centroids[index].weight / self.merged.centroids[other].weight).abs() < Constants::SMALL);

            let z: f64 = proposition * normalizer;
            let q0: f64 = cum / weight;
            let q2: f64 = (cum + proposition) / weight;

            let is_improvement: bool = (z <= (q0 * (1. - q0))) & (z <= (q2 * (1. - q2)));

            if is_improvement | is_too_close | is_too_small {
                self.merged.centroids[other].weight += self.merged.centroids[index].weight;
                self.merged.centroids[other].value += (
                    self.merged.centroids[index].value - self.merged.centroids[other].value
                ) * self.merged.centroids[index].weight / self.merged.centroids[other].weight + Constants::SMALL;
            }

            else {
                cum += self.merged.centroids[other].weight;
                other += 1;
                self.merged.centroids[other].weight = self.merged.centroids[index].weight;
                self.merged.centroids[other].value = self.merged.centroids[index].value;
            }

            if other != index {
                self.merged.centroids[index].reset();
            }

        }

        self.unmerged.centroids.clear();

        self.weight = cum;

        self.merged.centroids.retain(|centroid| centroid.weight > 0.);

        self.processed = true;

    }

    fn _cdf(&self, value: f64) -> Option<f64> {

        let mut k: f64 = 0.0;

        let mut index: usize = 0;

        let mut closest: Option<&Centroid> = None;

        for centroid in self.merged.centroids.iter() {

            closest = Some(centroid);
            
            if centroid.value >= value {
                break
            }
            
            k += centroid.weight;
            index += 1;
        }

        let centroid = match closest {
            Some(object) => object,
            None => return None,
        };


        if value == centroid.value {
            let mut tiebreak: f64 = centroid.weight;
            for neighbor in self.merged.centroids[index..].iter() {
                if neighbor.weight > centroid.weight {
                    break
                }
                else {
                    tiebreak += neighbor.weight
                }
            }

            return Some((k + (tiebreak * 0.5)) / self.weight)
        }

        else if value > centroid.value {
            return Some(1.0)
        }

        else if index == 0 {
            return Some(0.0)
        }

        else {
            let right: &Centroid = &self.merged.centroids[index];
            let left: &Centroid = &self.merged.centroids[index - 1];
    
            k -= left.weight * 0.5;
            let m: f64 = (right.value - left.value) / (left.weight * 0.5 + right.weight * 0.5);
            let x: f64 = (value - left.value) / m;
            
            Some((k + x) / self.weight)
        }
        
    }

    fn cdf(&mut self, value: f64) -> Option<f64> {

        if self._should_merge() {
            self.merge();
        }

        self._cdf(value)

    }

    fn cdfs(&mut self, values: Vec<f64>) -> Vec<Option<f64>> {

        if self._should_merge() {
            self.merge();
        }

        values.par_iter()
            .map(|value| { self._cdf(*value) })
            .collect()
    }

}

fn main() {

    let delta: f64 = 100.0;

    let mut digest: Digest = Digest::new(delta);

    digest.push(1.0, 1.0);
    digest.push(4.0, 1.0);
    digest.push(3.0, 1.0);
    digest.push(2.0, 1.0);
    digest.push(1.0, 1.0);
    digest.append(vec![1.0], 1.0);

    digest.merge();

    let _: Option<f64> = digest.cdf(1.0);
    let cdfs: Vec<Option<f64>> = digest.cdfs(vec![1.0]);

    println!("{:?}", cdfs)

}

// #[pymodule]
// fn tdrust(_py: Python, m: &PyModule) -> PyResult<()> {

//     m.add_class::<Centroid>()?;
//     m.add_class::<Container>()?;
//     m.add_class::<Digest>()?;

//     Ok(())
// }