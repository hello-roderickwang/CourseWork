% @Date    : 2019-10-12 17:15:20
% @Author  : Xuenan(Roderick) Wang
% @Email   : roderick_wang@outlook.com
% @GitHub  : https://github.com/hello-roderickwang

% Generation of a random matrix of rank equal to 3
A = rand(20, 5);
B = matrix_filling(A, 3);
disp('My random matrix of rank equal to 3 is:')
disp(B)
disp('Rank of this matrix is:');
disp(rank(B));

% Question (a)
B2 = put_random_zeros(B, 2);
B3 = put_random_zeros(B, 3);
B4 = put_random_zeros(B, 4);
B5 = put_random_zeros(B, 5);

[U2, S2, V2] = svd(B2);
[U3, S3, V3] = svd(B3);
[U4, S4, V4] = svd(B4);
[U5, S5, V5] = svd(B5);

disp('Rank of B2 matrix is:');
disp(rank(B2));
disp('Rank of B3 matrix is:');
disp(rank(B3));
disp('Rank of B4 matrix is:');
disp(rank(B4));
disp('Rank of B5 matrix is:');
disp(rank(B5));

% Question (b)
B2f = matrix_filling(B2, 3)
B3f = matrix_filling(B3, 3)
B4f = matrix_filling(B4, 3)
B5f = matrix_filling(B5, 3)

% Question (c)
var2 = var(var(B2f-B))
var3 = var(var(B3f-B))
var4 = var(var(B4f-B))
var5 = var(var(B5f-B))

function B = matrix_filling(A, target_rank)
    [U, S, V] = svd(A);
    B = U(:, 1:target_rank)*S(1:target_rank, 1:target_rank)*V(:,1:target_rank)';
end
    
function A = put_random_zeros(A, zero_num)
    for i = 1:zero_num
        A(randi(20), randi(5)) = 0;
    end
end