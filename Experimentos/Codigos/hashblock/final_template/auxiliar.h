#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned int bigendian(unsigned int num){

	return	(((num & 0xff000000) >> 24) | ((num & 0x00ff0000) >> 8) 
	   	  | ((num & 0x0000ff00) << 8) | (num << 24));

}



// we need a helper function to convert hex to binary, this function is unsafe and slow, but very readable (write something better)
void hex2bin(unsigned char* dest, unsigned char* src)
{
//	unsigned char bin;
	int c, pos;
	char buf[3];

	pos=0;
	c=0;
	buf[2] = 0;
	while(c < strlen((const char *)src))
	{
		// read in 2 characaters at a time
		buf[0] = src[c++];
		buf[1] = src[c++];
		// convert them to a interger and recast to a char (uint8)
		dest[pos++] = (unsigned char)strtol(buf, NULL, 16);
	}
	
}

// this function is mostly useless in a real implementation, were only using it for demonstration purposes
void hexdump(unsigned char* data, int len)
{
	int c;
	
	c=0;
	while(c < len)
	{
		printf("%.2x", data[c++]);
	}
	printf("\n");
}

// this function swaps the byte ordering of binary data, this code is slow and bloated (write your own)
void byte_swap(unsigned char* data, int len) {
	int c;
	unsigned char tmp[len];
	
	c=0;
	while(c<len)
	{
		tmp[c] = data[len-(c+1)];
		c++;
	}
	
	c=0;
	while(c<len)
	{
		data[c] = tmp[c];
		c++;
	}
}

/*
	Receives little endian target, returns
	hexadecimal array with target. This implements
	https://bitcoin.stackexchange.com/questions/44579/how-is-a-block-header-hash-compared-to-the-target-bits
*/

void bin2hex(unsigned int t, unsigned char* dest){
	
	// Get exponent
	unsigned int e = (t >> (8*0)) & 0xff;
	printf("Exponent: %d in hexa: 0x",e);
	hexdump((unsigned char*)&e,sizeof(e));

	// Get mantisa
	unsigned int c = (t) & 0xffffff00;
	printf("Mantisa %u in hexa: 0x", c);
	hexdump((unsigned char*)&c,sizeof(c));	

	// compute target = c * 2**(8*(e - 3))
	int shift = e;
	printf("Shift %d in hexa: 0x%x\n", shift, shift);
	// 
	int i=0;
	printf("Buffer size: %d\n",int(strlen((const char *)dest)));
	while(i<32)
		dest[i++] = 0;
	i = 0;
	while(i<3){
		dest[32-shift+i] = (c >> (8*(i+1))) & 0xff;
		printf("c: 0x%x   i: %d   ch: 0x%.2x\n", c,i,(c >> (8*(i+1))) & 0xff);
		i++;
	}
}



void target2hex(unsigned int target, unsigned char *htarget){

	// Process the target just for fun
	printf("Target: %u\n",bigendian(target));
	printf("Target in hexa, little endian: 0x");
	hexdump((unsigned char*)&target,sizeof(target));

	// Target should be in bigendian, so we reverse it
	unsigned int btarget = bigendian(target);
	printf("Target in hexa, big endian   : 0x");
	hexdump((unsigned char*)&btarget,sizeof(btarget));

	//unsigned char buf[32];
	bin2hex(btarget,htarget);

	printf("target %u in hexa: 0x", btarget);
	hexdump(htarget,32);	

}






