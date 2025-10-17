% Script: run_standalone.m
addpath('../AD_CTRL/');

dirAD = '../AD_CTRL/AD_s3/';
dirCTRL = '../AD_CTRL/CTRL_s3/';
tivPath = '../AD_CTRL/covariateADCTRLsexAgeTIV.csv';

% Chiama la funzione SENZA l'handle python
Preliminaries(dirAD, dirCTRL, tivPath);