set multi lay 1,2

set xlabel 'x'
set ylabel 'y'
plot 'wall.dat' u 1:2 ps 1 t ''

anglebin=pi/100
set xlabel 'angle'
set ylabel 'radius'
plot [-pi/2:pi/2] 'wall.dat' u (int($3/anglebin)*anglebin):(($1-256)**2+($2-256)**2) ps 1 smooth unique t 'binned angle'

unset multi
