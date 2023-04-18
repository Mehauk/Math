class Matrix {
  final int n;
  final int m;

  late final sorted x;

  List<List<num>> arr = [];

  Matrix(this.n, this.m) {
    arr = [
      [0]
    ];
    x = Sorter();
    x.c();
  }
}

class Sorter with sorted {
  final x = 0;
}

mixin sorted {
  void c() {
    print("cant x");
  }
}

// // this one is wierd
// mixin sorted2 on Sorter {
//   void c() {
//     print(x);
//   }
// }

extension sortist on Sorter {
  void c() {
    print(x);
  }
}

void main(List<String> args) {
  Matrix(1, 2);
}
