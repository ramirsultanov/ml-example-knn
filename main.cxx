#include "seaborn.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <tuple>
#include <cmath>
#include <set>

int main() {
  Seaborn s;
  Storage store;

  std::string dataset = "iris.csv";
  bool ret = s.loadData(dataset);

	map<string, Storage> args;
	store.setString("species");
	args["hue"] = store;

	string markers[3] = { "o", "s","D"};
	store.setStringArray(markers, 3);
	args["markers"] = store;

	bool rel = s.pairplot(args);

  std::string ppFile = "pairplot.png";
	s.saveGraph(ppFile);
  const auto pairplot = cv::imread(ppFile);
  cv::imshow("pairplot raw", pairplot);

  std::cout.flush();
  std::cout << std::endl;

  struct Row {
    float sepal_length;
    float sepal_width;
    float petal_length;
    float petal_width;
    std::string species;
  };
  std::vector<Row> rows;

  std::ifstream in(dataset);
  std::string line;
  if (!std::getline(in, line)) {
    return -1;
  }
  std::string top = line;
  while (std::getline(in, line)) {
    std::vector<std::string> values;
    auto end = line.find(',');
    decltype(end) start = 0;
    // int i = 0;
    while (end != std::string::npos) {
      values.push_back(line.substr(start, end - start));
      // std::cout << values[i++] << std::endl;
      start = end + 1;
      end = line.find(',', start);
    }
    if (values.size() != 4) {
      return -2;
    }
    std::string label = line.substr(start, end - start);
    rows.push_back(Row{
        std::stof(values[0]),
        std::stof(values[1]),
        std::stof(values[2]),
        std::stof(values[3]),
        label
    });
  }
  in.close();

  Row max{
      std::numeric_limits<float>::lowest(),
      std::numeric_limits<float>::lowest(),
      std::numeric_limits<float>::lowest(),
      std::numeric_limits<float>::lowest(),
      ""
  };
  for (const auto& row : rows) {
    if (row.sepal_length > max.sepal_length) {
      max.sepal_length = row.sepal_length;
    }
    if (row.sepal_width > max.sepal_width) {
      max.sepal_width = row.sepal_width;
    }
    if (row.petal_length > max.petal_length) {
      max.petal_length = row.petal_length;
    }
    if (row.petal_width > max.petal_width) {
      max.petal_width = row.petal_width;
    }
  }
  for (auto& row : rows) {
    std::apply(
      [&](auto... field) {
        ((row.*field /= max.*field), ...);
      },
      std::make_tuple(
        &Row::sepal_length,
        &Row::sepal_width,
        &Row::petal_length,
        &Row::petal_width
      )
    );
    // row.sepal_length /= max.sepal_length;
    // row.sepal_width /= max.sepal_width;
    // row.petal_length /= max.petal_length;
    // row.petal_width /= max.petal_width;
  }

  std::string datasetNormal = "iris.normal.csv";
  std::ofstream out(datasetNormal);
  out << top << "\n";
  // out << std::fixed << std::setprecision(1);
  for (const auto& row : rows) {
    out << row.sepal_length << "," << row.sepal_width << "," <<
        row.petal_length << "," << row.petal_width << "," << row.species <<
        "\n";
  }
  out.close();

  ret = s.loadData(datasetNormal);

	rel = s.pairplot(args);

  std::string ppNormalFile = "pairplot.normal.png";
	s.saveGraph(ppNormalFile);
  const auto pairplotNormal = cv::imread(ppNormalFile);
  cv::imshow("pairplot normal", pairplotNormal);
  cv::waitKey(1000);

  std::string learnFile = "iris.learn.csv";
  std::ofstream learn(learnFile);
  learn << top << "\n";
  std::string testFile = "iris.test.csv";
  std::ofstream test(testFile);
  test << top << "\n";

  std::map<std::string, std::size_t> count;
  for (const auto& row : rows) {
    count[row.species]++;
  }

  for (auto& c : count) {
    c.second -= 3;
  }

  std::map<std::string, std::size_t> cur;
  for (const auto& row : rows) {
    if (cur[row.species]++ < count[row.species]) {
      learn << row.sepal_length << "," << row.sepal_width << "," <<
          row.petal_length << "," << row.petal_width << "," << row.species <<
          "\n";
    } else {
      test << row.sepal_length << "," << row.sepal_width << "," <<
          row.petal_length << "," << row.petal_width << "," << row.species <<
          "\n";
    }
  }
  learn.close();
  test.close();

  ret = s.loadData(learnFile);

  auto n = 0u;
  for (const auto& c : count) {
    n += c.second;
  }
  auto k = static_cast<unsigned>(std::sqrt(n));
  std::cout << std::endl << "n: " << n << "\tk: " << k << std::endl;

  const auto distance = [](const Row& x, const Row& y) {
    const auto diff =
        (x.sepal_length - y.sepal_length) +
        (x.sepal_width - y.sepal_width) +
        (x.petal_length - y.petal_length) +
        (x.petal_width - y.petal_width);
    return diff * diff;
  };

  Row user;
  std::string input;

  std::cout << "enter <sepal_length sepal_width petal_length petal_width>" <<
      std::endl;

  std::cin >> input;
  user.sepal_length = std::stof(input);
  std::cin >> input;
  user.sepal_width = std::stof(input);
  std::cin >> input;
  user.petal_length = std::stof(input);
  std::cin >> input;
  user.petal_width = std::stof(input);
  using distance_type = std::invoke_result_t<decltype(distance), Row, Row>;
  using multiset_value_type = std::tuple<distance_type, std::string>;
  const auto comp = [](
      const multiset_value_type& lhs,
      const multiset_value_type& rhs
  ) {
    return std::get<distance_type>(lhs) < std::get<distance_type>(rhs);
  };
  std::multiset<multiset_value_type, decltype(comp)> distances(comp);
  for (const auto& row : rows) {
    distances.insert({distance(row, user), row.species});
  }

  std::map<std::string, std::size_t> res;
  {
    auto i = 0u;
    for (auto it = distances.begin(); it != distances.end() && i < k; it++, i++) {
      res[std::get<std::string>(*it)]++;
    }
  }

  std::tuple<std::string, std::size_t> result = {
      std::string(),
      std::numeric_limits<std::size_t>::min(),
  };
  for (const auto& r : res) {
    if (std::get<std::size_t>(result) < r.second) {
      result = { r.first, r.second, };
    }
  }
  std::cout << "result: " << std::get<std::string>(result) << std::endl;

  return 0;
}
