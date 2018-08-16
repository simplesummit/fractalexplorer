/* log.h -- defines logging utilities. I found this online, see comments below:

INCLUDED from the internet, adapted for this project, fractalrender.
original: https://github.com/rxi/log.c , distributed from MIT license

*/


/**
 * Copyright (c) 2017 rxi
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the MIT license. See `log.c` for details.
*/

#ifndef __LOG_H__
#define __LOG_H__

#include <stdarg.h>
#include <stdio.h>

typedef void (*log_LockFn)(void *udata, int lock);

#define LOG_FATAL     0
#define LOG_ERROR     1
#define LOG_WARN      2
#define LOG_INFO      3
#define LOG_DEBUG     4
#define LOG_TRACE     5

#define log_trace(...) log_log(LOG_TRACE, __FILE__, __LINE__, __VA_ARGS__)
#define log_debug(...) log_log(LOG_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define log_info(...)  log_log(LOG_INFO,  __FILE__, __LINE__, __VA_ARGS__)
#define log_warn(...)  log_log(LOG_WARN,  __FILE__, __LINE__, __VA_ARGS__)
#define log_error(...) log_log(LOG_ERROR, __FILE__, __LINE__, __VA_ARGS__)
#define log_fatal(...) log_log(LOG_FATAL, __FILE__, __LINE__, __VA_ARGS__)

void log_set_lock(log_LockFn fn);
int log_get_level();
void log_set_level(int level);
void log_set_quiet(int enable);

void log_log(int level, const char *file, int line, const char *fmt, ...);

#endif