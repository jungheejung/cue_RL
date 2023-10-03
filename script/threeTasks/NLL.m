

% AUTHOR: Aryan Yazdanpanh

function [negloglike]=NLL(xpar, Data, model_type,Noxious, High_Cue)
% % NLL %
%PURPOSE:  NLL calculation for parameter estimation, called by fit_fun().
%
%INPUT ARGUMENTS
%   xpar:       input parameters depending on the model type
%   Data:      Data for each subject
%   model_type:       'mdl1: one alpha, constant weight', 'mdl2: two alpha, constant weight',
% 'mdl3: one alpha, changing weight' ,'mdl4: two alpa, chanigng weight',
% 'mdl5: one alpha, changing weight (w on N)' ,'mdl6: two alpa, chanigng weight (w on N)'
%   Noxious: a vector of the Noxious inputs for different trials
%   High_Cue: a vector of cues (high (1) vs low(0)) for different trials


%OUTPUT ARGUMENTS
%   negloglike:      negaive log likelihood


%% defining the models:
num_trial=size(Data,1);  % number of trials
NLLs = zeros(1,num_trial);
NLLs_pain = zeros(1,num_trial);
NLLs_exp = zeros(1,num_trial);
negloglike=0;
rng('shuffle'); % random number generator shuffle
idx_ExpH = find(strcmp(Data.param_cue_type, 'high_cue'), 1);
idx_ExpL = find(strcmp(Data.param_cue_type, 'low_cue'), 1);
first_exp_high_cue = Data.event02_expect_angle(idx_ExpH);
first_exp_low_cue = Data.event02_expect_angle(idx_ExpL);


switch model_type
    case 'mdl1'
        % mdl1: one alpha, constant weight:
        % defining parameters:
        alpha=xpar(1);
        w=xpar(2);
        painerror=xpar(3);
        experror=xpar(4);
        % defingin expectations and pains:
        ExpectationH=first_exp_high_cue*ones(num_trial+1,1); % Expectation_highCue matrix
        ExpectationL=first_exp_low_cue*ones(num_trial+1,1); % Expectation_lowCue matrix
        Pain=zeros(num_trial+1,1); % Pain matrix
        PE=zeros(num_trial+1,1); % Pain predcition error matrix
        PainH=zeros(num_trial+1,1); % Pain matrix for high cues
        PainL=zeros(num_trial+1,1); % Pain matrix for low cues
        for ii=1:num_trial
            if High_Cue(ii)==1
                Pain(ii,1)=(1-w)*Noxious(ii,1)+w*ExpectationH(ii,1);
                PE(ii,1)=Pain(ii,1)-ExpectationH(ii,1);
                ExpectationH(ii+1,1)=ExpectationH(ii,1)+alpha*PE(ii,1);
                ExpectationL(ii+1,1)=ExpectationL(ii,1);
                PainH(ii,1)=Pain(ii,1);
                PainL(ii,1)=nan;
                NLLs_pain(ii) = -log(max(realmin,normpdf(Data.event04_actual_angle(ii),Pain(ii,1),painerror))); % pain log likelihood
                NLLs_exp(ii) = -log(max(realmin,normpdf(Data.event02_expect_angle(ii),ExpectationH(ii,1),experror))); % exp log likelihood
                NLLs(ii)=NLLs_pain(ii)+NLLs_exp(ii);
            else
                Pain(ii,1)=(1-w)*Noxious(ii,1)+w*ExpectationL(ii,1);
                PE(ii,1)=Pain(ii,1)-ExpectationL(ii,1);
                ExpectationL(ii+1,1)=ExpectationL(ii,1)+alpha*PE(ii,1);
                ExpectationH(ii+1,1)=ExpectationH(ii,1);
                PainL(ii,1)=Pain(ii,1);
                PainH(ii,1)=nan;
                NLLs_pain(ii) = -log(max(realmin,normpdf(Data.event04_actual_angle(ii),Pain(ii,1),painerror))); % pain log likelihood
                NLLs_exp(ii) = -log(max(realmin,normpdf(Data.event02_expect_angle(ii),ExpectationL(ii,1),experror))); % exp log likelihood
                NLLs(ii)=NLLs_pain(ii)+NLLs_exp(ii);
            end
            % calculating NLL (based on NLL_exp and NLL_pain)
            negloglike=negloglike+NLLs(ii);  % calculate log likelihood
        end


    case 'mdl2'
        % 'mdl2: two alpha, constant weight':
        % defining parameters:
        alpha_c=xpar(1);
        alpha_i=xpar(2);
        w=xpar(3);
        painerror=xpar(4);
        experror=xpar(5);
        % defingin expectations and pains:
        ExpectationH=first_exp_high_cue*ones(num_trial+1,1); % Expectation_highCue matrix
        ExpectationL=first_exp_low_cue*ones(num_trial+1,1); % Expectation_lowCue matrix
        Pain=zeros(num_trial+1,1); % Pain matrix
        PE=zeros(num_trial+1,1); % Pain predcition error matrix
        PainH=zeros(num_trial+1,1); % Pain matrix for high cues
        PainL=zeros(num_trial+1,1); % Pain matrix for low cues
        for ii=1:num_trial
            if High_Cue(ii)==1
                Pain(ii,1)=(1-w)*Noxious(ii,1)+w*ExpectationH(ii,1);
                PE(ii,1)=Pain(ii,1)-ExpectationH(ii,1);
                if PE(ii,1)>=0
                    alpha=alpha_c;
                else
                    alpha=alpha_i;
                end
                ExpectationH(ii+1,1)=ExpectationH(ii,1)+alpha*PE(ii,1);
                ExpectationL(ii+1,1)=ExpectationL(ii,1);
                PainH(ii,1)=Pain(ii,1);
                PainL(ii,1)=nan;
                NLLs_pain(ii) = -log(max(realmin,normpdf(Data.event04_actual_angle(ii),Pain(ii,1),painerror))); % pain log likelihood
                NLLs_exp(ii) = -log(max(realmin,normpdf(Data.event02_expect_angle(ii),ExpectationH(ii,1),experror))); % exp log likelihood
                NLLs(ii)=NLLs_pain(ii)+NLLs_exp(ii);
            else
                Pain(ii,1)=(1-w)*Noxious(ii,1)+w*ExpectationL(ii,1);
                PE(ii,1)=Pain(ii,1)-ExpectationL(ii,1);
                if PE(ii,1)>=0
                    alpha=alpha_i;
                else
                    alpha=alpha_c;
                end
                ExpectationL(ii+1,1)=ExpectationL(ii,1)+alpha*PE(ii,1);
                ExpectationH(ii+1,1)=ExpectationH(ii,1);
                PainL(ii,1)=Pain(ii,1);
                PainH(ii,1)=nan;
                NLLs_pain(ii) = -log(max(realmin,normpdf(Data.event04_actual_angle(ii),Pain(ii,1),painerror))); % pain log likelihood
                NLLs_exp(ii) = -log(max(realmin,normpdf(Data.event02_expect_angle(ii),ExpectationL(ii,1),experror))); % exp log likelihood
                NLLs(ii)=NLLs_pain(ii)+NLLs_exp(ii);
            end
            % calculating sum of squared errors (based on the sum of errors for Pain and expectation)
            negloglike=negloglike+NLLs(ii);  % calculate log likelihood
        end


    case 'mdl3'
        % 'mdl3: one alpha, changing weight':
        % defining parameters:
        alpha=xpar(1);
        gamma=xpar(2);
        painerror=xpar(3);
        experror=xpar(4);
        % defingin expectations and pains:
        ExpectationH=first_exp_high_cue*ones(num_trial+1,1); % Expectation_highCue matrix
        ExpectationL=first_exp_low_cue*ones(num_trial+1,1); % Expectation_lowCue matrix
        Pain=zeros(num_trial+1,1); % Pain matrix
        PE=zeros(num_trial+1,1); % Pain predcition error matrix
        PainH=zeros(num_trial+1,1); % Pain matrix for high cues
        PainL=zeros(num_trial+1,1); % Pain matrix for low cues
        for ii=1:num_trial
            if High_Cue(ii)==1
                w=2/(1+exp(gamma*abs(Noxious(ii,1)-ExpectationH(ii,1))));
                Pain(ii,1)=(1-w)*Noxious(ii,1)+w*ExpectationH(ii,1);
                PE(ii,1)=Pain(ii,1)-ExpectationH(ii,1);
                ExpectationH(ii+1,1)=ExpectationH(ii,1)+alpha*PE(ii,1);
                ExpectationL(ii+1,1)=ExpectationL(ii,1);
                PainH(ii,1)=Pain(ii,1);
                PainL(ii,1)=nan;
                NLLs_pain(ii) = -log(max(realmin,normpdf(Data.event04_actual_angle(ii),Pain(ii,1),painerror))); % pain log likelihood
                NLLs_exp(ii) = -log(max(realmin,normpdf(Data.event02_expect_angle(ii),ExpectationH(ii,1),experror))); % exp log likelihood
                NLLs(ii)=NLLs_pain(ii)+NLLs_exp(ii);
            else
                w=2/(1+exp(gamma*abs(Noxious(ii,1)-ExpectationL(ii,1))));
                Pain(ii,1)=(1-w)*Noxious(ii,1)+w*ExpectationL(ii,1);
                PE(ii,1)=Pain(ii,1)-ExpectationL(ii,1);
                ExpectationL(ii+1,1)=ExpectationL(ii,1)+alpha*PE(ii,1);
                ExpectationH(ii+1,1)=ExpectationH(ii,1);
                PainL(ii,1)=Pain(ii,1);
                PainH(ii,1)=nan;
                NLLs_pain(ii) = -log(max(realmin,normpdf(Data.event04_actual_angle(ii),Pain(ii,1),painerror))); % pain log likelihood
                NLLs_exp(ii) = -log(max(realmin,normpdf(Data.event02_expect_angle(ii),ExpectationL(ii,1),experror))); % exp log likelihood
                NLLs(ii)=NLLs_pain(ii)+NLLs_exp(ii);
            end
            % calculating sum of squared errors (based on the sum of errors for Pain and expectation)
            negloglike=negloglike+NLLs(ii);  % calculate log likelihood
        end


    case 'mdl4'
%         'mdl4: two alpa, chanigng weight':
        % defining parameters:
        alpha_c=xpar(1);
        alpha_i=xpar(2);
        gamma=xpar(3);
        painerror=xpar(4);
        experror=xpar(5);
        % defingin expectations and pains:
        ExpectationH=first_exp_high_cue*ones(num_trial+1,1); % Expectation_highCue matrix
        ExpectationL=first_exp_low_cue*ones(num_trial+1,1); % Expectation_lowCue matrix
        Pain=zeros(num_trial+1,1); % Pain matrix
        PE=zeros(num_trial+1,1); % Pain predcition error matrix
        PainH=zeros(num_trial+1,1); % Pain matrix for high cues
        PainL=zeros(num_trial+1,1); % Pain matrix for low cues
        for ii=1:num_trial
            if High_Cue(ii)==1
                w=2/(1+exp(gamma*abs(Noxious(ii,1)-ExpectationH(ii,1))));
                Pain(ii,1)=(1-w)*Noxious(ii,1)+w*ExpectationH(ii,1);
                PE(ii,1)=Pain(ii,1)-ExpectationH(ii,1);
                if PE(ii,1)>=0
                    alpha=alpha_c;
                else
                    alpha=alpha_i;
                end
                ExpectationH(ii+1,1)=ExpectationH(ii,1)+alpha*PE(ii,1);
                ExpectationL(ii+1,1)=ExpectationL(ii,1);
                PainH(ii,1)=Pain(ii,1);
                PainL(ii,1)=nan;
                NLLs_pain(ii) = -log(max(realmin,normpdf(Data.event04_actual_angle(ii),Pain(ii,1),painerror))); % pain log likelihood
                NLLs_exp(ii) = -log(max(realmin,normpdf(Data.event02_expect_angle(ii),ExpectationH(ii,1),experror))); % exp log likelihood
                NLLs(ii)=NLLs_pain(ii)+NLLs_exp(ii);
            else
                w=2/(1+exp(gamma*abs(Noxious(ii,1)-ExpectationL(ii,1))));
                Pain(ii,1)=(1-w)*Noxious(ii,1)+w*ExpectationL(ii,1);
                PE(ii,1)=Pain(ii,1)-ExpectationL(ii,1);
                if PE(ii,1)>=0
                    alpha=alpha_i;
                else
                    alpha=alpha_c;
                end
                ExpectationL(ii+1,1)=ExpectationL(ii,1)+alpha*PE(ii,1);
                ExpectationH(ii+1,1)=ExpectationH(ii,1);
                PainL(ii,1)=Pain(ii,1);
                PainH(ii,1)=nan;
                NLLs_pain(ii) = -log(max(realmin,normpdf(Data.event04_actual_angle(ii),Pain(ii,1),painerror))); % pain log likelihood
                NLLs_exp(ii) = -log(max(realmin,normpdf(Data.event02_expect_angle(ii),ExpectationL(ii,1),experror))); % exp log likelihood
                NLLs(ii)=NLLs_pain(ii)+NLLs_exp(ii);
            end
            % calculating sum of squared errors (based on the sum of errors for Pain and expectation)
            negloglike=negloglike+NLLs(ii);  % calculate log likelihood
        end


 case 'mdl5'
        % 'mdl5: one alpha, changing weight': w on N
        % defining parameters:
        alpha=xpar(1);
        gamma=xpar(2);
        painerror=xpar(3);
        experror=xpar(4);
        % defingin expectations and pains:
        ExpectationH=first_exp_high_cue*ones(num_trial+1,1); % Expectation_highCue matrix
        ExpectationL=first_exp_low_cue*ones(num_trial+1,1); % Expectation_lowCue matrix
        Pain=zeros(num_trial+1,1); % Pain matrix
        PE=zeros(num_trial+1,1); % Pain predcition error matrix
        PainH=zeros(num_trial+1,1); % Pain matrix for high cues
        PainL=zeros(num_trial+1,1); % Pain matrix for low cues
        for ii=1:num_trial
            if High_Cue(ii)==1
                w=2/(1+exp(gamma*abs(Noxious(ii,1)-ExpectationH(ii,1))));
                Pain(ii,1)=w*Noxious(ii,1)+(1-w)*ExpectationH(ii,1);
                PE(ii,1)=Pain(ii,1)-ExpectationH(ii,1);
                ExpectationH(ii+1,1)=ExpectationH(ii,1)+alpha*PE(ii,1);
                ExpectationL(ii+1,1)=ExpectationL(ii,1);
                PainH(ii,1)=Pain(ii,1);
                PainL(ii,1)=nan;
                NLLs_pain(ii) = -log(max(realmin,normpdf(Data.event04_actual_angle(ii),Pain(ii,1),painerror))); % pain log likelihood
                NLLs_exp(ii) = -log(max(realmin,normpdf(Data.event02_expect_angle(ii),ExpectationH(ii,1),experror))); % exp log likelihood
                NLLs(ii)=NLLs_pain(ii)+NLLs_exp(ii);
            else
                w=2/(1+exp(gamma*abs(Noxious(ii,1)-ExpectationL(ii,1))));
                Pain(ii,1)=w*Noxious(ii,1)+(1-w)*ExpectationL(ii,1);
                PE(ii,1)=Pain(ii,1)-ExpectationL(ii,1);
                ExpectationL(ii+1,1)=ExpectationL(ii,1)+alpha*PE(ii,1);
                ExpectationH(ii+1,1)=ExpectationH(ii,1);
                PainL(ii,1)=Pain(ii,1);
                PainH(ii,1)=nan;
                NLLs_pain(ii) = -log(max(realmin,normpdf(Data.event04_actual_angle(ii),Pain(ii,1),painerror))); % pain log likelihood
                NLLs_exp(ii) = -log(max(realmin,normpdf(Data.event02_expect_angle(ii),ExpectationL(ii,1),experror))); % exp log likelihood
                NLLs(ii)=NLLs_pain(ii)+NLLs_exp(ii);
            end
            % calculating sum of squared errors (based on the sum of errors for Pain and expectation)
            negloglike=negloglike+NLLs(ii);  % calculate log likelihood
        end


    case 'mdl6'
%         'mdl6: two alpa, chanigng weight': w on N
        % defining parameters:
        alpha_c=xpar(1);
        alpha_i=xpar(2);
        gamma=xpar(3);
        painerror=xpar(4);
        experror=xpar(5);
        % defingin expectations and pains:
        ExpectationH=first_exp_high_cue*ones(num_trial+1,1); % Expectation_highCue matrix
        ExpectationL=first_exp_low_cue*ones(num_trial+1,1); % Expectation_lowCue matrix
        Pain=zeros(num_trial+1,1); % Pain matrix
        PE=zeros(num_trial+1,1); % Pain predcition error matrix
        PainH=zeros(num_trial+1,1); % Pain matrix for high cues
        PainL=zeros(num_trial+1,1); % Pain matrix for low cues
        for ii=1:num_trial
            if High_Cue(ii)==1
                w=2/(1+exp(gamma*abs(Noxious(ii,1)-ExpectationH(ii,1))));
                Pain(ii,1)=w*Noxious(ii,1)+(1-w)*ExpectationH(ii,1);
                PE(ii,1)=Pain(ii,1)-ExpectationH(ii,1);
                if PE(ii,1)>=0
                    alpha=alpha_c;
                else
                    alpha=alpha_i;
                end
                ExpectationH(ii+1,1)=ExpectationH(ii,1)+alpha*PE(ii,1);
                ExpectationL(ii+1,1)=ExpectationL(ii,1);
                PainH(ii,1)=Pain(ii,1);
                PainL(ii,1)=nan;
                NLLs_pain(ii) = -log(max(realmin,normpdf(Data.event04_actual_angle(ii),Pain(ii,1),painerror))); % pain log likelihood
                NLLs_exp(ii) = -log(max(realmin,normpdf(Data.event02_expect_angle(ii),ExpectationH(ii,1),experror))); % exp log likelihood
                NLLs(ii)=NLLs_pain(ii)+NLLs_exp(ii);
            else
                w=2/(1+exp(gamma*abs(Noxious(ii,1)-ExpectationL(ii,1))));
                Pain(ii,1)=w*Noxious(ii,1)+(1-w)*ExpectationL(ii,1);
                PE(ii,1)=Pain(ii,1)-ExpectationL(ii,1);
                if PE(ii,1)>=0
                    alpha=alpha_i;
                else
                    alpha=alpha_c;
                end
                ExpectationL(ii+1,1)=ExpectationL(ii,1)+alpha*PE(ii,1);
                ExpectationH(ii+1,1)=ExpectationH(ii,1);
                PainL(ii,1)=Pain(ii,1);
                PainH(ii,1)=nan;
                NLLs_pain(ii) = -log(max(realmin,normpdf(Data.event04_actual_angle(ii),Pain(ii,1),painerror))); % pain log likelihood
                NLLs_exp(ii) = -log(max(realmin,normpdf(Data.event02_expect_angle(ii),ExpectationL(ii,1),experror))); % exp log likelihood
                NLLs(ii)=NLLs_pain(ii)+NLLs_exp(ii);
            end
            % calculating sum of squared errors (based on the sum of errors for Pain and expectation)
            negloglike=negloglike+NLLs(ii);  % calculate log likelihood
        end




    otherwise
        error('Unexpected model type')
end



