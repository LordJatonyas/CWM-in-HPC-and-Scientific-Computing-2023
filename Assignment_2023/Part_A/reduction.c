#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define NUM_ELS 100000

int main() {
	
    float random_array[NUM_ELS];
    float sum=0;

    # pragma omp parallel default(none) private(i, x, sum) shared(random_array)
    {
    # pragma omp for reduction(+:x)
    for(int i=0; i<NUM_ELS; i++) {
        float x = ((float)rand())/((float)RAND_MAX);
        random_array[i] = x;
    }

    # pragma omp for reduction(+:sum)
    for(int i=0; i<NUM_ELS; i++) {
        sum+=random_array[i];
    }
    }

    printf("\nAverage:\t%f\n", sum/(float)NUM_ELS);

    return(0);
}
