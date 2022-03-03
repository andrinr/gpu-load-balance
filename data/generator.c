#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    int count = atoi(argv[1]);
    int dimensions = atoi(argv[2]);

    FILE *fp;
    fp = fopen("uniform.dat", "w+");

	for (int i = 0; i < count; i++){
        float p = (float)rand()/(float)(RAND_MAX) - 0.5;
		for (int d = 0; d < dimensions; d++){
            fprintf(fp, "%f, ", p);
        }
        fprintf(fp, "\n");
	}

	fclose(fp);
}