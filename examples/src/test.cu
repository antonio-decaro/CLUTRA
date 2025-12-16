#include <clutra.hpp>
#include <iostream>

int main() {
  std::cout << "CLUTRA Library Test" << std::endl;
  clutra::frontier::FrontierMLB<uint32_t> frontier(1024);


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
  clutra::profile::KernelProfilerManager::instance().printSummary();
  return 0;
}