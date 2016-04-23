% 10601A/SV-F15: Introduction to Machine Learning
% Programming Assignment 4: HMM for Speech Recognition
%
% TASK 1: Write a routine that obtains the forward variables.
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
% ============================================================
% OUTPUT  alphas[num_states, num_observations]: The forward variables

function [alphas] = get_forward_variables(observations, params)
    num_observations = size(observations, 1);
    num_states = size(params.initial_probs, 1);
    alphas = zeros(num_states, num_observations);  
    
    % Get the observations for state 1, 2, and 3
    O(:,1) = mvnpdf(observations, params.observation_probs_means{1}, params.observation_probs_covariances{1});
    O(:,2)= mvnpdf(observations, params.observation_probs_means{2}, params.observation_probs_covariances{2});
    O(:,3) = mvnpdf(observations, params.observation_probs_means{3}, params.observation_probs_covariances{3});
    
    %Get the initial alphas for each state
    alphas(1,1) = params.initial_probs(1) * O(1,1);
    alphas(2,1) = params.initial_probs(2) * O(1,2);
    alphas(3,1) = params.initial_probs(3) * O(1,3);
    
    % Get the rest of the alphas
    for i = 2:num_observations
      % Get observations for state 1
      for j = 1:num_states
        alphas(1,i) = alphas(1,i) + O(i,1) * alphas(j,i-1) * params.transition_probs(j,1);
      end
      % Get observations for state 2
      for j = 1:num_states
        alphas(2,i) = alphas(2,i) + O(i,2) * alphas(j,i-1) * params.transition_probs(j,2);
      end
      % Get observations for state 3
      for j = 1:num_states
        alphas(3,i) = alphas(3,i) + O(i,3) * alphas(j,i-1) * params.transition_probs(j,3);
      end
    end
    alphas
end
