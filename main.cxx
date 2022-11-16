#include "seaborn.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <tuple>
#include <cmath>
#include <set>

struct Row {
  float sepal_length;
  float sepal_width;
  float petal_length;
  float petal_width;
  std::string species;
};

std::tuple<std::vector<Row>, std::string> readCsv(const std::string& dataset) {
  std::vector<Row> rows;
  std::ifstream in(dataset);
  std::string line;
  if (!std::getline(in, line)) {
    return {};
  }
  std::string top = line;
  while (std::getline(in, line)) {
    std::vector<std::string> values;
    auto end = line.find(',');
    decltype(end) start = 0;
    while (end != std::string::npos) {
      values.push_back(line.substr(start, end - start));
      start = end + 1;
      end = line.find(',', start);
    }
    if (values.size() != 4) {
      return {};
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
  return {rows, top};
}

std::string predict(
    std::size_t k,
    const std::vector<Row>& rows,
    const Row& input
) {
  const auto distance = [](const Row& x, const Row& y) {
    const auto diff =
        (x.sepal_length - y.sepal_length) +
        (x.sepal_width - y.sepal_width) +
        (x.petal_length - y.petal_length) +
        (x.petal_width - y.petal_width);
    return diff * diff;
  };

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
    distances.insert({distance(row, input), row.species});
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

  return std::get<std::string>(result);
}

std::size_t test(
    std::size_t k,
    const std::vector<Row>& rows,
    const std::vector<Row>& testRows
) {
  std::vector<std::string> results;
  for (const auto& testRow : testRows) {
    results.push_back(predict(k, rows, testRow));
  }

  std::size_t matches = 0;
  for (auto i = 0u; i < testRows.size(); i++) {
    if (testRows[i].species == results[i]) {
      matches++;
    }
  }

  return matches;
}

int main() {
  Seaborn s;
  Storage store;

  const std::string dataset = "iris.csv";
  bool ret = s.loadData(dataset);

	map<string, Storage> args;
	store.setString("species");
	args["hue"] = store;

	string markers[3] = { "o", "s","D"};
	store.setStringArray(markers, 3);
	args["markers"] = store;

	bool rel = s.pairplot(args);

  const std::string ppFile = "pairplot.png";
	s.saveGraph(ppFile);
  const auto pairplot = cv::imread(ppFile);
  cv::imshow("pairplot raw", pairplot);

  std::cout.flush();
  std::cout << std::endl;

  auto [rows, topStr] = readCsv(dataset);
  const auto& top = topStr;

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

  const std::string datasetNormal = "iris.normal.csv";
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

  const std::string ppNormalFile = "pairplot.normal.png";
	s.saveGraph(ppNormalFile);
  const auto pairplotNormal = cv::imread(ppNormalFile);
  cv::imshow("pairplot normal", pairplotNormal);
  cv::waitKey(1000);

  const std::string learnFile = "iris.learn.csv";
  std::ofstream learnStream(learnFile);
  learnStream << top << "\n";
  const std::string testFile = "iris.test.csv";
  std::ofstream testStream(testFile);
  testStream << top << "\n";

  std::map<std::string, std::size_t> count;
  for (const auto& row : rows) {
    count[row.species]++;
  }

  constexpr auto learnCoef = 0.9f;
  auto learnCount = count;
  for (auto& c : learnCount) {
    c.second *= learnCoef;
  }

  // auto testCount = count;
  // for (auto& c : testCount) {
    // c.second = count.at(c.first) - learnCount.at(c.first);
  // }

  // for (auto& c : count) {
    // c.second -= 3;
  // }

  std::map<std::string, std::size_t> cur;
  for (const auto& row : rows) {
    if (cur[row.species]++ < learnCount[row.species]) {
      learnStream << row.sepal_length << "," << row.sepal_width << "," <<
          row.petal_length << "," << row.petal_width << "," << row.species <<
          "\n";
    } else {
      testStream << row.sepal_length << "," << row.sepal_width << "," <<
          row.petal_length << "," << row.petal_width << "," << row.species <<
          "\n";
    }
  }
  learnStream.close();
  testStream.close();

  // ret = s.loadData(learnFile);

  // auto n = 0u;
  // for (const auto& c : count) {
  //   n += c.second;
  // }
  // auto k = static_cast<unsigned>(std::sqrt(n));
  // std::cout << std::endl << "n: " << n << "\tk: " << k << std::endl;

  const auto [testRows, dummy] = readCsv(testFile);

  auto maxMatchesK = std::make_tuple(test(1u, rows, testRows), std::size_t(1));
  std::cout << std::endl <<
      "k: " << 1 << "\tmatches: " << std::get<0>(maxMatchesK) << std::endl;
  for (auto i = 2u; i < rows.size(); i++) {
    const auto matches = test(i, rows, testRows);
    std::cout << std::endl <<
        "k: " << i << "\tmatches: " << matches << std::endl;
    if (matches >= std::get<0>(maxMatchesK)) {
      std::get<0>(maxMatchesK) = matches;
      std::get<1>(maxMatchesK) = i;
    }
  }
  const auto& k = std::get<1>(maxMatchesK);
  std::cout << std::endl <<
      "k: " << k << std::endl;

  Row user;
  std::string input;

  std::cout <<
      "enter <sepal_length sepal_width petal_length petal_width>" << std::endl;

  std::cin >> input;
  user.sepal_length = std::stof(input);
  std::cin >> input;
  user.sepal_width = std::stof(input);
  std::cin >> input;
  user.petal_length = std::stof(input);
  std::cin >> input;
  user.petal_width = std::stof(input);

  std::cout << "result: " << predict(k, rows, user) << std::endl;

  return 0;
}
