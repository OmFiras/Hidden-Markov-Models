% 10601A/SV-F15: Introduction to Machine Learning
% Programming Assignment 4: HMM for Speech Recognition
%
% TASK 2: Write a routine that obtains the backward variables.
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
% OUTPUT  betas[num_states, num_observations]: The backward variables

function [betas] = get_backward_variables(observations, params)
  num_observations = size(observations, 1);
  num_states = size(params.initial_probs, 1);
  betas = zeros(num_states, num_observations);

  % Get the observations for state 1, 2, and 3
  O(:,1) = mvnpdf(observations, params.observation_probs_means{1}, params.observation_probs_covariances{1});
  O(:,2)= mvnpdf(observations, params.observation_probs_means{2}, params.observation_probs_covariances{2});
  O(:,3) = mvnpdf(observations, params.observation_probs_means{3}, params.observation_probs_covariances{3});
  
  % Get your initial betas
  betas(1,num_observations) = 1;
  betas(2,num_observations) = 1;
  betas(3,num_observations) = 1;
  
  % Get the rest of the betas
  for i = num_observations-1:-1:1
    % Get observations for state 1
    for j = 1:num_states
      betas(1,i) = betas(1,i) + betas(j,i+1) * params.transition_probs(1,j) * O(i+1,j);
    end
    % Get observations for state 2
    for j = 1:num_states
      betas(2,i) = betas(2,i) + betas(j,i+1) * params.transition_probs(2,j) * O(i+1,j);
    end
    % Get observations for state 3
    for j = 1:num_states
      betas(3,i) = betas(3,i) + betas(j,i+1) * params.transition_probs(3,j) * O(i+1,j);
    end
  end
end
