use network::{activation::SIGMOID, Network, learning::{DataSet, Data}};

fn main() {
    let mut cnt = 0;
    let mut net = Network::new(vec![5, 5, 1], 0.02, 0.3, SIGMOID);
    //  net.load(&"network.json".to_string());
    for i in 0..30 {
        let tests = DataSet::from_table(vec![
            (vec![0, 1, 0, 1, 0], vec![1]),
            (vec![0, 1, 1, 1, 0], vec![1]),
            (vec![0, 1, 0, 0, 0], vec![0]),
            (vec![1, 0, 0, 0, 1], vec![1]),
            (vec![0, 0, 0, 1, 1], vec![0]),
            (vec![1, 0, 0, 0, 0], vec![0]),
            (vec![1, 1, 0, 0, 1], vec![0]),
            (vec![1, 1, 0, 1, 0], vec![0]),
        ]);

        net.train(tests, 1000, false);

        let res = net.predict(Data::new(vec![1, 0, 1, 0, 1]));
        let Data(data) = &res;
        if data.iter().map(|x| x.round() as i32).collect::<Vec<i32>>() == vec![0] {
            cnt += 1;
        }
        
        res.view();
    }
    net.save(&"network.json".to_string());
    println!("{} / {}", cnt, 100);
}
