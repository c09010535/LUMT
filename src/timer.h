#ifndef TIMER_H_INCLUDED
#define TIMER_H_INCLUDED

#include <time.h>

#ifdef _WIN32

/*for windows*/
typedef struct tagSTimer
{
	__int64 freq;
	__int64 start;
	__int64 stop;
} STimer;

#else

/*for linux*/
#include <sys/time.h>
typedef struct tagSTimer
{
	struct timespec start;
	struct timespec stop;
} STimer;

#endif

int TimerInit(STimer *tmr);
int TimerStart(STimer *tmr);
int TimerStop(STimer *tmr);
double TimerGetRuntime(STimer *tmr);

#endif