#include "utils.hpp"
#include <clutra.hpp>
#include <iostream>

int main(int argc, char** argv) {

  Options opts;
  CLI::App app{"CLUTRA example"};
  auto cli_handles = configureBaseCLI(app, opts);
  CLI11_PARSE(app, argc, argv);
  finalizeGraphOptions(opts, cli_handles);

  std::cerr << "[*] Reading CSR" << std::endl;
  clutra::graph::Properties properties;
  auto csr = readCSR<float, uint32_t, uint32_t>(opts, &properties);
  auto graph = clutra::graph::createGraph(csr, properties);
  std::cerr << "[*] CSR read complete" << std::endl;
  printGraphInfo(graph);
  printStealingOptions(opts);

  clutra::frontier::FrontierMLB<uint32_t> frontier(1024);

  int device_id;
  cudaGetDevice(&device_id);

  std::cout << clutra::detail::device::getDeviceName(device_id) << std::endl;
  std::cout << "Num SMs: " << clutra::detail::device::getNumSMs(device_id) << std::endl;

  if (frontier.empty()) {
    std::cout << "Frontier is initially empty." << std::endl;
  } else {
    std::cout << "Frontier is NOT empty." << std::endl;
  }

  frontier.insert(10);
  if (frontier.check(10)) {
    std::cout << "Element 10 is in the frontier." << std::endl;
  } else {
    std::cout << "Element 10 is NOT in the frontier." << std::endl;
  }
  frontier.insert(11);
  frontier.insert(512);

  std::cout << frontier.size() << " elements in the frontier." << std::endl;
  frontier.computeActiveFrontier();
  std::cout << "Active frontier size: " << frontier.getActiveFrontierSize() << std::endl;

  frontier.remove(10);
  if (!frontier.check(10)) {
    std::cout << "Element 10 has been removed from the frontier." << std::endl;
  } else {
    std::cout << "Element 10 is still in the frontier." << std::endl;
  }

  if (frontier.empty()) {
    std::cout << "Frontier is now empty." << std::endl;
  } else {
    std::cout << "Frontier is NOT empty." << std::endl;
  }

  std::cout << frontier.size() << " elements in the frontier." << std::endl;
  clutra::profile::KernelProfilerManager::instance().printSummary(opts.profiling_detail);
  return 0;
}
