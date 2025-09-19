@echo off
echo Running deepLP example...
poetry run deeplp --batches 1 --batch_size 32 --iterations 100 --case 1 --example 1 --do_plot
pause
