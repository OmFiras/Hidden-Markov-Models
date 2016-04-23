% 10601A/SV-F15: Introduction to Machine Learning
% Programming Assignment 4: HMM for Speech Recognition
%
% TASK 4: Write a method for the maximization step and return new parameters.
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
%         xis[num_states, num_states, num_observations]
%         gammas[num_states, num_observations]
% ============================================================
% OUTPUT new_params - object similar to params.

function [new_params] = maximization_step(observations, params, xis, gammas)
  num_features = size(observations, 2);
  num_observations = size(observations, 1);
  num_states = size(params.initial_probs, 1);

  new_params = params;
  o = observations;
  
  % get the new initial_probs
  for s = 1:num_states
    new_params.initial_probs(s) = gammas(s,1);
  end
  
  % Get the new transition_probs
  for s = 1:num_states
    for s_dash = 1:num_states;
      new_params.transition_probs(s, s_dash) = (sum(xis(s, s_dash,:)) - xis(s,s_dash,num_observations)) / (sum(gammas(s,:)) - gammas(s,num_observations));
    end
  end
  
  % Get the new means
  for s = 1:num_states
    nom = 0;
    den = 0;
    for t = 1:num_observations
      nom = nom + gammas(s,t) * o(t, :);
      den = den + gammas(s,t);
    end
    new_params.observation_probs_means{s} = nom / den;
  end
  
  % Get the new covariance matrix
  for s = 1:num_states
    nom = 0;
    den = 0;
    for t = 1:num_observations
      nom = nom + gammas(s,t) * o(t,:)' * o(t,:);
      den = den + gammas(s,t);
    end
    new_params.observation_probs_covariances{s} = nom / den - (new_params.observation_probs_means{s})' * new_params.observation_probs_means{s};
  end
end
