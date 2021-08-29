// author: zlz
#pragma once

#ifndef NOCOPY
#define NOCOPY(Class)     \
 private:                       \
  Class(const Class&) = delete; \
  Class& operator=(const Class&) = delete
#endif
