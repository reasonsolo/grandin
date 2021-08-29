// author: zlz
#pragma once

#include "vendor/DogFood/DogFood.hpp"

#define MONITOR_COUNTER(name, value) \
  DogFood::Send(DogFood::Metric(#name, value, DogFood::Counter))

#define MONITOR_TIMER(name, value) \
  DogFood::Send(DogFood::Metric(#name, value, DogFood::Timer))

#define MONITOR_GAUGE(name, value) \
  DogFood::Send(DogFood::Metric(#name, value, DogFood::Gauge))

#define MONITOR_HISTO(name, value) \
  DogFood::Send(DogFood::Metric(#name, value, DogFood::Histogram))

