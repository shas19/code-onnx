import numpy as np
import math

scale_in = -11
H = 4
W = 6

#create random array of size 4*6
a = np.random.uniform(-2, 2, [4,6]).tolist()
b = np.zeros([4,6]).tolist()

float_ans = 0

for h in range(H):
	for w in range(W):
		b[h][w] = int(a[h][w]*(1<<(-(scale_in))))  

# b is 16 bit max
sum_square = 0
sum_square_float = 0

for h in range(H):
	for w in range(W):
		sum_square += (b[h][w]>>8)*(b[h][w]>>8) # should have a different downscale?
		# print(b[h][w]//int(1<<8))
		# print(sum_square)
		sum_square_float += (a[h][w]*a[h][w])

sum_square_scale = 2*(scale_in + 8) 
# sum_square might overflow 16 bits..

# y = 1/sqrt(sum_square)

print("sum square values")
print(sum_square_float)
print(sum_square*(2**sum_square_scale))



y_scale = -7
y_low = 1
y_high = (1<<9)-1

cmp_val_scale = sum_square_scale + 2*y_scale

while(y_low+1 < y_high):
	y_mid = ((y_low + y_high)>>1);
	cmp_val = y_mid*y_mid*sum_square
	if (cmp_val*(2**cmp_val_scale) > 1):
		y_high = y_mid
	else:
		y_low = y_mid

# y is 8 bits with scale -7 
y = y_low

print("\ninv norm values")
y_float = 1/math.sqrt(sum_square_float)
print(y_float)
print(y*(2**y_scale))

# now calculate the final output
out_float = np.zeros([4,6]).tolist()
out = np.zeros([4,6]).tolist()
out_scale = scale_in+8+y_scale

for h in range(H):
	for w in range(W):
		out[h][w] = (b[h][w]>>8)*y
		out_float[h][w] = a[h][w]*y_float
		# print(out_float[h][w], out[h][w]*(2**out_scale))

# Calculating the error
out = np.array(out)*(2**out_scale)
out_float = np.array(out_float)

# mean absolute error
mae = np.mean(np.abs(out - out_float))
print("Mean Absolute Error: " + str(mae))

# mean relative error
mre = np.mean(np.abs(np.divide(np.abs(out - out_float), out_float)))
print("Mean Relative Error: " + str(mre))




# // A = Normalise(A)
# void NormaliseL2(MYINT* A, MYINT N, MYINT H, MYINT W, MYINT C, MYINT scaleA, MYINT shrA) {
# 	for (MYITE n = 0; n < N; n++) {
# 		for (MYITE c = 0; c < C; c++) {

# 			// calculate the sum square
# 			int32_t sumSquare = 0;
# 			MYINT shrAdiv = (1<<shrA);

# 			for (MYITE h = 0; h < H; h++) {
# 				for (MYITE w = 0; w < W; w++) {
# 					MYINT tmp = (A[n * H * W * C + h * W * C + w * C + c] / shrAdiv);
# 					sumSquare += tmp*tmp;
# 				}
# 			}

# 			cout <<"scaleA: "<< scaleA<<endl;
# 			cout <<"shrA: "<<shrA<<endl;

# 			cout << "sumSquare " << endl;
# 			cout << sumSquare << endl;
# 			cout << (2*scaleA + 2*shrA) << endl;

# 			// calculate the inverse square roor of sumSquare
# 			MYINT yLow = 1;

# 			// yHigh: A number of length shrA with all 1s in binary representation e.g. for shrA=8 --> y_high = 0b11111111
# 			MYINT yHigh = (1<<shrA - 1);   

# 			// one: value of 1 with same scale as y*y*sumSquare
# 			// scale of sumSquare = 2*scaleA + 2*shrA
# 			// since we assume scale of y = 1 - shrA
# 			// scale of y*y*sumSquare =  2*scale_in + 2*shrA + 2(1-shrA) = 2*scale_in + 2
# 			int32_t one = ( 1<< (-(2*scaleA + 2)) ); 

# 			// binary search for the inverse square root 
# 			while( yLow+1 < yHigh){

# 				// using int64_t sotherwise (y*y*sumSquare) will overflow
# 				int64_t yMid = ((yHigh + yLow)>>1);

# 				int64_t cmpValue = (yMid*yMid)*sumSquare;

# 				// cout << "cmp: "<< cmpValue << endl;

# 				if(cmpValue > one){
# 					yHigh = yMid;	
# 				}	
# 				else {
# 					yLow = yMid;
# 				}
# 			}
# 			MYINT inverseNorm = yLow;

# 			// cout << "Inverse Nomrm" << endl;
# 			// cout << inverseNorm << endl;
# 			// cout << (1-shrA) << endl;

# 			// cout << "input before" << endl;
# 			// cout << A[n * H * W * C + c] << endl;
# 			// cout << scaleA << endl;

# 			// multiply all elements by the 1/sqrt(sumSquare)
# 			for (MYITE h = 0; h < H; h++) {
# 				for (MYITE w = 0; w < W; w++) {
# 					A[n * H * W * C + h * W * C + w * C + c]  = (A[n * H * W * C + h * W * C + w * C + c]  / shrAdiv)*inverseNorm;  
# 				}
# 			}
			
# 			cout << A[n * H * W * C + c] << endl;	
# 			cout << scaleA+1 << endl;
# 		}

# 	}
# 	return;
# }