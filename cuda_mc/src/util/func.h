/*
 * func.h
 *
 *  Created on: 02-Feb-2009
 *      Author: alee
 */

#ifndef FUNC_H_
#define FUNC_H_

#define XFUNC(x,y) x ## _ ## y
#define FUNC(x, y) XFUNC(x, y)

#include <cuda_runtime.h>

#define PI 3.14159265358979f

#define logsumexp(a,b) ( ((a)>(b)) ? ((a)+log(1+exp((b)-(a)))) : ((b)+log(1+exp((a)-(b)))) )

#endif /* FUNC_H_ */
