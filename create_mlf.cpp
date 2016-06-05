#include <iostream>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <stdint.h>

int main(int argc,char* argv[])
{
	FILE *fp=NULL,*sp=NULL,*lp=NULL;
	int r,breadth;
	char filename[2000];
	char sound[2000];

	fp = fopen(argv[1],"r");
	sp = fopen(argv[2],"w");
	fprintf(sp,"#!MLF!#\n");
	while((r=fscanf(fp,"%s",filename))!=EOF)
	{
printf("%s\n",filename);
		fprintf(sp,"\"%s\"\n",filename);
		lp = fopen(filename,"r");
		while((r=fscanf(lp,"%s",sound))!=EOF)
			fprintf(sp,"%s\n",sound);
		fprintf(sp,".\n");
		fclose(lp);
	}	
	fclose(fp);
	fclose(sp);
}

