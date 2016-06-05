#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
using namespace std;
int main(int argc, char* argv[])
{
    FILE *proto=NULL;
    int no_of_states,mix,initstate=2,i,j,initmix;
    float  value[2000];
    char targetkind[2000];
	int final_feature_length;
	int num_of_mixtures;
	char filename[2000];

	if(argc!=4)
	{
		printf("Usage: <exe> <filename> <vecsize> <no_mixtures>\n");
		exit(0);
	}

	final_feature_length = atoi(argv[2]);
	num_of_mixtures = atoi(argv[3]);
	strcpy(filename,argv[1]);
    no_of_states = 3;

    strcpy(targetkind,"USER");
    proto= fopen(filename,"w");
    fprintf(proto,"~o <VecSize> %d <%s> \n",final_feature_length,targetkind);
    
    fprintf(proto,"~h \"%s\" \n",filename);
    fprintf(proto,"<BeginHMM> \n");
    fprintf(proto,"<VecSize> %d <%s> \n",final_feature_length,targetkind);
    fprintf(proto,"<NumStates> %d ",no_of_states);
    for(mix=1;mix<=num_of_mixtures;mix++)
	value[mix]=(1/(float)(num_of_mixtures));
    while(initstate<=(no_of_states-1))
    {
        initmix=1;
	fprintf(proto,"\n<State> %d <NumMixes> %d ",initstate,num_of_mixtures);
	while(initmix<=num_of_mixtures)
	{
	    if(num_of_mixtures!=1)	    
		fprintf(proto,"\n<Mixture> %d %f \n",initmix,value[initmix]);
	    fprintf(proto,"<Mean> %d \n",final_feature_length);
	    for(i=1;i<=final_feature_length;i++)
	    {
		fprintf(proto,"0.0 ");
	    }
	    fprintf(proto,"\n<Variance> %d \n",final_feature_length);
	    for(i=1;i<=final_feature_length;i++)
	    {
		fprintf(proto,"1.0 ");
	    }
	    initmix++;
	}
	initstate++;
    }
    fprintf(proto,"\n<TransP> %d \n",no_of_states);
    for(i=1;i<=no_of_states;i++)
    {
	for(j=1;j<=no_of_states;j++)
	{
	    if((j==1)||(i==(no_of_states)))
	    {
		fprintf(proto,"0.0\t");
	    }
	    else
	    {
		fprintf(proto,"%1.2f \t ",(1/(float)(no_of_states-1)));
	    }
	}
	fprintf(proto,"\n");
	
    }
    fprintf(proto,"<EndHMM>\n");
    fclose(proto);
}


