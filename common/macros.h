// author: zlz
#ifndef COMMON_MACROS_H_
#define COMMON_MACROS_H_

#ifndef NOCOPY
#define NOCOPY(Class)     \
 private:                       \
  Class(const Class&) = delete; \
  Class& operator=(const Class&) = delete
#endif

#endif