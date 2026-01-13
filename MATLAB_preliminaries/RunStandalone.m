% Script: run_standalone.m
addpath('../CMEPDA_project_2024/AD_CTRL/');

dirAD = '../CMEPDA_project_2024/AD_CTRL/AD_s3/';
dirCTRL = '../CMEPDA_project_2024/AD_CTRL/CTRL_s3/';
tivPath = '../CMEPDA_project_2024/AD_CTRL/covariateADCTRLsexAgeTIV.csv';

% Chiama la funzione SENZA l'handle python
Preliminaries(dirAD, dirCTRL, tivPath);