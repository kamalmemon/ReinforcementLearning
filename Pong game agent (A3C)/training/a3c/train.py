import torch
import torch.nn.functional as F
import torch.optim as optim

from envs import ObsNorm
from model import ActorCritic
from pong import Pong
from simple_ai import PongAi

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, counter, lock, optimizer=None):
    torch.manual_seed(args.seed + rank)
    env = Pong(headless = False)
    # env.seed(args.seed + rank)
    opponent = PongAi(env,2)
    model = ActorCritic(1, 3)
    model.train()
    
    obsNorm = ObsNorm()
    state = obsNorm.prepro(env.reset()[0])
    
    state = torch.from_numpy(state)
    done = True
    game_done = 0

    episode_length = 0
    while True:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []
        for step in range(args.num_steps):
            episode_length += 1
            inputTensor = state.unsqueeze(0); 
            value, logit, (hx, cx) = model((inputTensor,
                                            (hx, cx)))
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)
            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)
            action2 = opponent.get_action()
            (state, obs2), (reward, reward2), done, info = env.step((action.numpy(), action2))
            if done:
                game_done +=1
            done = game_done == 19 or episode_length >= args.max_episode_length
            reward = max(min(reward, 10), -10)
            state = obsNorm.prepro(state)
            with lock:
                counter.value += 1  
            if done:
               # print('episode_length')
               # print(episode_length)
                episode_length = 0
                obsNorm.reset()
                state = obsNorm.prepro(env.reset()[0])
            if game_done == 19:
                game_done = 0
            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((state.unsqueeze(0), (hx, cx)))
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + args.gamma * \
                values[i + 1] - values[i]
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]

        optimizer.zero_grad()

        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()
