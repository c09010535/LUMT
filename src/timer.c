#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif
#include "timer.h"

int TimerInit(STimer *tmr)
{
	if (NULL == tmr)
	{
		return -1;
	}

#ifdef _WIN32
	tmr->start = 0;
	tmr->stop = 0;
	return (QueryPerformanceFrequency((LARGE_INTEGER *)&(tmr->freq))==0 ? -1 : 0);
#else
	tmr->start.tv_sec = 0;
	tmr->start.tv_nsec = 0;
	tmr->stop.tv_sec = 0;
	tmr->stop.tv_nsec = 0;
	return 0;
#endif
}

int TimerStart(STimer *tmr)
{
	if (NULL == tmr)
	{
		return -1;
	}

#ifdef _WIN32
	return (QueryPerformanceCounter((LARGE_INTEGER *)&(tmr->start))==0 ? -1 : 0);
#else
#ifdef CLOCK_MONOTONIC
	return clock_gettime(CLOCK_MONOTONIC, &(tmr->start));
#else
	return clock_gettime(CLOCK_REALTIME, &(tmr->start));
#endif
#endif
}

int TimerStop(STimer *tmr)
{
	if (NULL == tmr)
	{
		return -1;
	}

#ifdef _WIN32
	return (QueryPerformanceCounter((LARGE_INTEGER *)&(tmr->stop))==0 ? -1 : 0);
#else
#ifdef CLOCK_MONOTONIC
	return clock_gettime(CLOCK_MONOTONIC, &(tmr->stop));
#else
	return clock_gettime(CLOCK_REALTIME, &(tmr->stop));
#endif
#endif
}

double TimerGetRuntime(STimer *tmr)
{
	if (NULL == tmr)
	{
		return -1.0;
	}

#ifdef _WIN32
	return ((double)(tmr->stop)-(double)(tmr->start))/(tmr->freq);
#else
	return (tmr->stop.tv_sec-tmr->start.tv_sec) + \
		(tmr->stop.tv_nsec-tmr->start.tv_nsec)/1.e+9;
#endif
}