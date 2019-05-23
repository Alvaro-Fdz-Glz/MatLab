%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  LAB EXERCISE 2. SPEAKER IDENTIFICATION
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Configuration variables (***TO BE COMPLETED***)
nspk = 16;       % Number of speakers  
fs = 16000;         % Sampling frequency
wst = 0.03;        % Window size (seconds)
fpt = 0.015;        % Frame period (seconds) 
ws = wst*fs;         % Window size (samples)
fp = fpt*fs;         % Frame period (samples)
ngauss = 8;     % Number of gaussians in the GMM models

% Other configuration variables
nbands = 40;   % Number of filters in the filterbank
numcep = 20;
options = statset('MaxIter',500);
frequency_warp = ["mel","bark","htkmel","fcmel"];

% Lists of training and testing speech files
nomlist_train = 'list_train.txt';
nomlist_test{1} = 'list_test1.txt';
nomlist_test{2} = 'list_test2.txt';

sr = 16000;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  TRAINING STAGE
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


val_gauss=zeros(1,2);
for i=1:10
    x=Calc(nspk,fs,wst,fpt,ws,fp,i,nbands,numcep,options,frequency_warp,nomlist_train,nomlist_test,sr)
    val_gauss(i,:)=x;
end

val_cep=zeros(1,2);
for i=1:25
    x=Calc(nspk,fs,wst,fpt,ws,fp,7,nbands,i,options,frequency_warp,nomlist_train,nomlist_test,sr)
    val_cep(i,:)=x;
end


function accuracy=Calc(nspk,fs,wst,fpt,ws,fp,ngauss,nbands,numcep,options,frequency_warp,nomlist_train,nomlist_test,sr)
% Speaker GMM models building
for i=1:nspk
    
    % Loading of the training files for speaker "i"
    %(***TO BE COMPLETED***)
    x=load_train_data(nomlist_train,i);
        
    % Feature extraction
    [cepstra,aspectrum,pspectrum]=melfcc(x,sr,'numcep',numcep,'wintime',wst, 'hoptime',fpt,'nbands',nbands,'fbtype',frequency_warp(1));
    x_processed=transpose(cepstra);
    %(***TO BE COMPLETED***)
    
    % Speaker GMM models building for speaker "i"
    x_glm{i}=fitgmdist(x_processed,ngauss,'CovarianceType','diagonal','Options',options);
    
    clearvars x_processed
end  % for i=1:nspk

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  TEST STAGE
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Speaker identification process

accuracy = zeros(2,1);
for t=1:size(nomlist_test,2)
       
   % Reading of the list containing the speech test files 
   fid = fopen(nomlist_test{t});
   
   if fid < 0
      fprintf('File %s does not exist\n', nomlist_test{t});
      return
   end
   info_test = textscan(fid, '%s%f');
   nfiles_test = length(info_test{1}); % number of test files
   actual_spk = int16(info_test{2});   % speaker label of each test file
   fclose(fid);
   fcst_spk = int16(length(info_test{1}));
   

   % Loop for each test file
   for k=1:nfiles_test
      % Loading of the test files
      fname_test = info_test{1}{k};  % name of the test file
      
      % Reading of the wav test file "fname_test"
      %(***TO BE COMPLETED***)
      x=audioread(fname_test);

      % Feature extraction
      %(***TO BE COMPLETED***)
      [cepstra,aspectrum,pspectrum]=melfcc(x,sr,'numcep',numcep,'wintime',wst, 'hoptime',fpt,'nbands',nbands);
      x_processed=transpose(cepstra);

      % Log-likelihood computation of each model for the current test file
      %(***TO BE COMPLETED***)
      res = zeros(nspk,1);
      for i=1:nspk
          res(i)=sum(log(pdf(x_glm{i},x_processed)));        
      end
      
      % Selection of the identified speaker
      %(***TO BE COMPLETED***)
      [M,fcst_spk(k)] = max(res,[],1);
 
   end  % for k=1:nfiles_test
   
   % Computation of the identification accuracy
   %(***TO BE COMPLETED***)
   cnfMatrix= confusionmat(actual_spk,fcst_spk); 
   accuracy(t) = sum(diag(cnfMatrix))/sum(cnfMatrix(:));  
end % t=1:size(nomlist_test,2),
end



   


   

