// author: zlz
#pragma once

#define NOCOPY(Class)           \
 private:                       \
  Class(const Class&) = delete; \
  Class& operator=(const Class&) = delete

#define SINGLETON(Class)                   \
 private:                                  \
  Class() = default;                       \
  Class(const Class&) = delete;            \
  Class& operator=(const Class&) = delete; \
                                           \
 public:                                   \
  static Class& GetInstance() {            \
    static Class instance;                 \
    return instance;                       \
  }
