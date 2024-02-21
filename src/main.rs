use network::{activation::SIGMOID, Network, learning::{DataSet, Data}};

fn main() {
    let mut net = Network::new(vec![4, 5, 3, 3], 1.0, SIGMOID);

    let tests = DataSet::from_table(vec![
        (vec![0, 0, 0, 0], vec![0, 0, 0]),
        (vec![0, 0, 0, 1], vec![0, 0, 1]),
        (vec![0, 0, 1, 0], vec![0, 1, 0]),
        (vec![0, 0, 1, 1], vec![0, 1, 1]),
        (vec![0, 1, 0, 0], vec![0, 0, 1]),
        (vec![0, 1, 0, 1], vec![0, 1, 0]),
        (vec![0, 1, 1, 0], vec![0, 1, 1]),
        (vec![0, 1, 1, 1], vec![1, 0, 0]),
        (vec![1, 0, 0, 0], vec![0, 1, 0]),
        (vec![1, 0, 0, 1], vec![0, 1, 1]),
        (vec![1, 0, 1, 0], vec![1, 0, 0]),
        // (vec![1, 0, 1, 1], vec![1, 0, 1]),
        (vec![1, 1, 0, 0], vec![0, 1, 1]),
        (vec![1, 1, 0, 1], vec![1, 0, 0]),
        (vec![1, 1, 1, 0], vec![1, 0, 1]),
        (vec![1, 1, 1, 1], vec![1, 1, 0]),
    ]);

    net.train(tests, 100000);

    net.predict(Data::new(vec![1, 0, 1, 1])).error();
}
