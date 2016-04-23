% 10601A/SV-F15: Introduction to Machine Learning
% Programming Assignment 4: HMM for Speech Recognition
%
% TASK 3: Write a method for the expectation step and return the variables.
% ============================================================
% INPUT
%       observations[num_observations, num_features]: a matrix where each row is an observation in the sequence.
%       params:
%         params.initial_probs[num_states, 1]: a column vector where row is a scalar
%             representing the initial probability of the state.
%         params.transition_probs[num_states, num_states]: a matrix where entry (i,j) represents the
%             probability of transitioning from state i to state j.
%         params.observation_probs_means[{i} => [1, num_features]]: a cell array where the ith element
%             is the mean vector of the observation probability distribution
%             of the ith state
%         params.observation_probs_covariances[{i} => [num_features, num_features]]: a cell array where the ith element
%             is the covariance matrix of the observation probability distribution
%             of the ith state;
%         alphas[num_states, num_observations]: The foward variables
%         betas[num_states, num_observations]: The backward variables
% ============================================================
% OUTPUT  xis[num_states, num_states, num_observations] -  xis[s,s',num_observations] is zero.
% OUTPUT  gammas[num_states, num_observations]

function [xis, gammas] = expectation_step(observations, params, alphas, betas)
  num_features = size(observations, 2);
  num_observations = size(observations, 1);
  num_states = size(params.initial_probs, 1);

  gammas = zeros(num_states, num_observations);
  gamma_variable = zeros(num_states, num_observations);
  xis = zeros(num_states, num_states, num_observations); % Let the matrix for the last time step be all 0's

  % Calculate the probability of a specific sequence occurance 
  P = 0;
  for i = 1:num_states
    P = P + alphas(i, num_observations);
  end
  
  % Get the observations for state 1, 2, and 3
  O(:,1) = mvnpdf(observations, params.observation_probs_means{1}, params.observation_probs_covariances{1});
  O(:,2)= mvnpdf(observations, params.observation_probs_means{2}, params.observation_probs_covariances{2});
  O(:,3) = mvnpdf(observations, params.observation_probs_means{3}, params.observation_probs_covariances{3});
  
  % Calculate the gammas
  for i = 1:num_observations
    % Get gammas for state 1
    gammas(1,i) = ( alphas(1,i) * betas(1,i) ) / P;
    % Get gammas for state 2
    gammas(2,i) = ( alphas(2,i) * betas(2,i) ) / P;
    % Get gammas for state 3
    gammas(3,i) = ( alphas(3,i) * betas(3,i) ) / P;
  end
  
  % Calculate the Xis
  for t = 1:num_observations-1 % t
    for s = 1:num_states % s
      for s_dash = 1:num_states % s'
        xis(s,s_dash,t) = ( alphas(s,t) * O(t+1, s_dash) * params.transition_probs(s,s_dash) * betas(s_dash,t+1) ) / P;
      end
    end
  end
end
