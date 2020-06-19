import numpy as np
import math

scale_in = -5
H = 4
W = 6

#create random array of size 4*6
a = np.random.uniform(0, 2, [4,6]).tolist()
b = a

float_ans = 0

for h in range(H):
	for w in range(W):
		b[h][w] = int(a[h][w]*(1<<(-(scale_in))))  


# b is 16 bit max
sum_square = 0
sum_square_float = 0

for h in range(H):
	for w in range(W):
		sum_square += (b[h][w]/(1<<8))*(b[h][w]/(1<<8))
		sum_square_float = (a[h][w]*a[h][w])			

sum_square_scale = 2*(scale_in+8) 
# sum_square might overflow 16 bits..

# y = 1/sqrt(sum_square)

print("sum square values")
print(sum_square_float)
print(sum_square/(1<<sum_square_scale))


y_float = 1/math.sqrt(sum_square_float)
print(y_float)

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

y = y_low

print(y*(2**y_scale))

# now calculate the output

