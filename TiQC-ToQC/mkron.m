function m = mkron(L)
m=1;
for j=1:length(L)
    m=kron(m,L{j});
end
end