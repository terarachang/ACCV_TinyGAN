import pdb
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import random
import utils
from model import CMPDisLoss

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
import functools
tfgan = tf.contrib.gan
from numpy_fid import numpy_calculate_frechet_distance as numpy_fid
from inception_tf import get_inception_score

class Solver(object):

	def __init__(self, train_loader, test_loader, real_loader, config):
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.real_loader = real_loader
		self.z_dim = config.z_dim
		self.c_dim = config.c_dim
		self.image_size = config.image_size
		self.g_conv_dim = config.g_conv_dim
		self.d_conv_dim = config.d_conv_dim
		self.g_repeat_num = config.g_repeat_num
		self.d_repeat_num = config.d_repeat_num
		self.lambda_gan = config.lambda_gan

		self.batch_size = config.batch_size
		self.num_epoch = config.num_epoch
		self.lr_decay_start = config.lr_decay_start
		self.g_lr = config.g_lr
		self.d_lr = config.d_lr
		self.n_critic = config.n_critic
		self.resume_epoch = config.resume_epoch

		# Miscellaneous.
		self.use_tensorboard = config.use_tensorboard
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.use_numpy_fid = config.use_numpy_fid

		# Directories.
		self.log_dir = config.log_dir
		self.sample_dir = config.sample_dir
		self.model_save_dir = config.model_save_dir
		self.result_dir = config.result_dir
		self.real_incep_stat_dir = config.real_incep_stat_dir
		self.real_fid_stat_dir = config.real_fid_stat_dir

		# Step size.
		self.log_step = config.log_step
		self.sample_step = config.sample_step
		self.model_save_step = config.model_save_step

		# Build the model and tensorboard.
		self.KDLoss = CMPDisLoss()
		self.G, self.D, self.g_optimizer, self.d_optimizer = utils.build_model(config)

		if self.use_tensorboard:
			self.logger = utils.build_tensorboard(self.log_dir)

	'''
	def gradient_penalty(self, y, x):
		"""Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
		weight = torch.ones(y.size()).to(self.device)
		dydx = torch.autograd.grad(outputs=y,
								   inputs=x,
								   grad_outputs=weight,
								   retain_graph=True,
								   create_graph=True,
								   only_inputs=True)[0]

		dydx = dydx.view(dydx.size(0), -1)
		dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
		return torch.mean((dydx_l2norm-1)**2)
	'''
	def dis_hinge(self, dis_real, dis_fake):
		d_loss_real = torch.mean(torch.relu(1. - dis_real))
		d_loss_fake = torch.mean(torch.relu(1. + dis_fake))
		return d_loss_real, d_loss_fake


	def gen_hinge(self, dis_fake):
		return -torch.mean(dis_fake)


	def train(self):
		loss = {}
		nrow = min(int(np.sqrt(self.batch_size)), 8)
		n_samples = nrow * nrow
		iter_per_epoch = len(self.train_loader.dataset) // self.batch_size
		max_iteration = self.num_epoch * iter_per_epoch
		lambda_l1 = 0.2
		print('Start training...')
		for epoch in tqdm(range(self.resume_epoch, self.num_epoch)):
			for i, (x_real, noise, label) in enumerate(tqdm(self.train_loader)):
				
				# lr decay
				if epoch * iter_per_epoch + i >= self.lr_decay_start:
					utils.decay_lr(self.g_optimizer, max_iteration, self.lr_decay_start, self.g_lr)
					utils.decay_lr(self.d_optimizer, max_iteration, self.lr_decay_start, self.d_lr)
					if i % 1000 == 0:
						print('d_lr / g_lr is updated to {:.8f} / {:.8f} !'
							.format(self.d_optimizer.param_groups[0]['lr'], self.g_optimizer.param_groups[0]['lr']))
				
				x_real = x_real.to(self.device)
				noise = noise.to(self.device)
				label = label.to(self.device)
				#'''
				# =================================================================================== #
				#							  1. Train the discriminator							  #
				# =================================================================================== #
				for param in self.D.parameters():
					param.requires_grad = True
				
				dis_real, real_list = self.D(x_real, label)
				real_list = [h.detach() for h in real_list]

				x_fake = self.G(noise, label).detach()
				dis_fake, _ = self.D(x_fake, label)

				d_loss_real, d_loss_fake = self.dis_hinge(dis_real, dis_fake)
				
				# sample
				try:
					x_real2, label2 = next(real_iter)
				except:
					real_iter = iter(self.real_loader)
					x_real2, label2 = next(real_iter)
				x_real2 = x_real2.to(self.device)
				label2 = label2.to(self.device)

				noise2 = torch.FloatTensor(utils.truncated_normal(self.batch_size*self.z_dim)) \
										.view(self.batch_size, self.z_dim).to(self.device)
#				 noise2 = torch.randn(self.batch_size, self.z_dim).to(self.device)
				dis_real2, _ = self.D(x_real2, label2)
				x_fake2 = self.G(noise2, label2).detach()
				dis_fake2, _ = self.D(x_fake2, label2)
				d_loss_real2, d_loss_fake2 = self.dis_hinge(dis_real2, dis_fake2)

				# Backward and optimize.
				d_loss = d_loss_real + d_loss_fake + 0.2*(d_loss_real2 + d_loss_fake2)
				
				self.d_optimizer.zero_grad()
				d_loss.backward()
				self.d_optimizer.step()

				# Logging.
				loss['D/loss_real'] = d_loss_real.item()
				loss['D/loss_fake'] = d_loss_fake.item()
				loss['D/loss_real2'] = d_loss_real2.item()
				loss['D/loss_fake2'] = d_loss_fake2.item()
				
				# =================================================================================== #
				#								2. Train the generator								  #
				# =================================================================================== #
				#'''

				x_fake = self.G(noise, label)
				
				for param in self.D.parameters():
					param.requires_grad = False
					
				dis_fake, fake_list = self.D(x_fake, label)
 
				g_loss_feat = self.KDLoss(real_list, fake_list) 
				g_loss_pix = F.l1_loss(x_fake, x_real)
				g_loss = g_loss_feat + lambda_l1 * g_loss_pix
				loss['G/loss_ft'] = g_loss_feat.item()
				loss['G/loss_l1'] = g_loss_pix.item()

				if (i+1) % self.n_critic == 0:
					dis_fake, _ = self.D(x_fake, label)
					g_loss_fake = self.gen_hinge(dis_fake)

					g_loss += self.lambda_gan * g_loss_fake
					
					# sample
					noise2 = torch.FloatTensor(utils.truncated_normal(self.batch_size*self.z_dim)) \
										.view(self.batch_size, self.z_dim).to(self.device)
#					 noise2 = torch.randn(self.batch_size, self.z_dim).to(self.device)
					x_fake2 = self.G(noise2, label2)
					dis_fake2, _ = self.D(x_fake2, label2)
					g_loss_fake2 = self.gen_hinge(dis_fake2)
					g_loss += 0.2 * self.lambda_gan * g_loss_fake2

					loss['G/loss_fake'] = g_loss_fake.item()
					loss['G/loss_fake2'] = g_loss_fake2.item()
				
				self.g_optimizer.zero_grad()
				g_loss.backward()
				self.g_optimizer.step()
				
				
				# =================================================================================== #
				#								  3. Miscellaneous									  #
				# =================================================================================== #

				# Print out training information.
				if (i+1) % self.log_step == 0:
					log = "[{}/{}]".format(epoch, i)
					for tag, value in loss.items():
						log += ", {}: {:.4f}".format(tag, value)
					print(log)

					if self.use_tensorboard:
						for tag, value in loss.items():
							self.logger.scalar_summary(tag, value, i+1)

			if epoch == 0 or (epoch+1) % self.sample_step == 0:
				with torch.no_grad():
					"""
					# randomly sampled noise
					noise = torch.FloatTensor(utils.truncated_normal(n_samples*self.z_dim)) \
										.view(n_samples, self.z_dim).to(self.device)
					label = label[:nrow].repeat(nrow)

					#label = np.random.choice(1000, nrow, replace=False)
					#label = torch.tensor(label).repeat(10).to(self.device)
					x_sample = self.G(noise, label)
					sample_path = os.path.join(self.sample_dir, '{}-sample.png'.format(epoch+1))
					save_image(utils.denorm(x_sample.cpu()), sample_path, nrow=nrow, padding=0)
					"""
					# recons
					n = min(x_real.size(0), 8)
					comparison = torch.cat([x_real[:n], x_fake[:n]])
					sample_path = os.path.join(self.sample_dir, '{}-train.png'.format(epoch+1))
					save_image(utils.denorm(comparison.cpu()), sample_path)
					print('Save fake images into {}...'.format(sample_path))
					
					# noise2
					comparison = torch.cat([x_real2[:n], x_fake2[:n]])
					sample_path = os.path.join(self.sample_dir, '{}-random.png'.format(epoch+1))
					save_image(utils.denorm(comparison.cpu()), sample_path)
					print('Save fake images into {}...'.format(sample_path))

					# noise sampled from BigGAN's test set
					try:
						x_real, noise, label = next(test_iter)
					except:
						test_iter = iter(self.test_loader)
						x_real, noise, label = next(test_iter)
					noise = noise.to(self.device)
					label = label.to(self.device)
					
					x_fake = self.G(noise, label).detach().cpu()
					n = min(x_real.size(0), 8)
					comparison = torch.cat([x_real[:n], x_fake[:n]])
					sample_path = os.path.join(self.sample_dir, '{}-test.png'.format(epoch+1))
					save_image(utils.denorm(comparison.cpu()), sample_path)
					print('Save fake images into {}...'.format(sample_path))					
			
			lambda_l1 = max(0.00, lambda_l1-0.01)
			# Save model checkpoints.
			if (epoch+1) % self.model_save_step == 0:
				utils.save_model(self.model_save_dir, epoch+1, self.G, self.D, self.g_optimizer, self.d_optimizer)


	def build_fid_graph(self):
		INCEPTION_FINAL_POOL = 'pool_3:0'
		INCEPTION_DEFAULT_IMAGE_SIZE = 299
		ACTIVATION_DIM = 2048

		def inception_activations(images, height=INCEPTION_DEFAULT_IMAGE_SIZE, width=INCEPTION_DEFAULT_IMAGE_SIZE, num_splits = 1):
			images = tf.image.resize_bilinear(images, [height, width])
			generated_images_list = array_ops.split(images, num_or_size_splits = num_splits)
			activations = tf.map_fn(
				fn = functools.partial(tfgan.eval.run_inception, output_tensor = INCEPTION_FINAL_POOL),
				elems = array_ops.stack(generated_images_list),
				parallel_iterations = 1,
				back_prop = False,
				swap_memory = True,
				name = 'RunClassifier')
			activations = array_ops.concat(array_ops.unstack(activations), 0)
			return activations

		self.images_holder = tf.placeholder(tf.float32, [None, 128, 128, 3])
		self.activations = inception_activations(self.images_holder)
		self.real_acts = tf.placeholder(tf.float32, [None, ACTIVATION_DIM], name = 'real_activations')
		self.fake_acts = tf.placeholder(tf.float32, [None, ACTIVATION_DIM], name = 'fake_activations')
		self.fid = tfgan.eval.frechet_classifier_distance_from_activations(self.real_acts, self.fake_acts)


	def test_intra_fid_all(self):
		test_num = 5000
		assert test_num % self.batch_size == 0, "test_num mod batch_size != 0"

		config = tf.ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = 0.5
		self.build_fid_graph()
		sess = tf.Session(config=config)
		sess.run(tf.global_variables_initializer())
		sess.graph.finalize()

#		 noises = np.load('noises.npy') # fixed noise from truncated normal

		fake_act = np.zeros((test_num, 2048))
		fid_scores = []
		with torch.no_grad():
			for c_id in tqdm(range(398)):
				label = torch.LongTensor(self.batch_size).fill_(c_id).to(self.device)
				for i in tqdm(range(0, test_num, self.batch_size)):
#					 noise = torch.FloatTensor(noises[i:i+self.batch_size]).to(self.device)
					# randomly sample
					noise = torch.FloatTensor(utils.truncated_normal(self.batch_size*self.z_dim)) \
										.view(self.batch_size, self.z_dim).to(self.device)
					x_sample = self.G(noise, label)
					
					if i == 0:
						sample_path = os.path.join(self.result_dir, '{}-test.png'.format(c_id))
						save_image(utils.denorm(x_sample[:64]), sample_path, nrow=8)
					x_sample = np.transpose(x_sample.cpu().numpy(), (0, 2, 3, 1)) # [NCHW] -> [NHWC]
					fake_act[i:i+self.batch_size] = sess.run(self.activations, {self.images_holder: x_sample})
#				np.save('fake_act_{}.npy'.format(c_id), fake_act)	  
				real_act = np.load(os.path.join(self.real_incep_stat_dir, 'act_{}.npy').format(c_id))
				
				if self.use_numpy_fid:
					real_mean, real_cov = np.mean(real_act, axis=0), np.cov(real_act, rowvar=False)
					fake_mean, fake_cov = np.mean(fake_act, axis=0), np.cov(fake_act, rowvar=False)
					fid = numpy_fid(fake_mean, fake_cov, real_mean, real_cov)
				else:
					fid = sess.run(self.fid, {self.real_acts: real_act, self.fake_acts: fake_act})

				print('[{}] FID: {:.3f}'.format(c_id, fid))
				fid_scores.append(fid)

		np.save(os.path.join(self.model_save_dir, 'intra_fid_scores_real_small.npy'), fid_scores)
		print('[TinyGAN] Intra-class FID: {:.3f}, std: {:.3f}'.format(np.mean(fid_scores), np.std(fid_scores)))
		sess.close()


	def test_inter_fid(self):
		test_num = 50000
		assert test_num % self.batch_size == 0, "test_num mod batch_size != 0"

		config = tf.ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = 0.5
		self.build_fid_graph()
		sess = tf.Session(config=config)
		sess.run(tf.global_variables_initializer())
		sess.graph.finalize()

		# load dumped mean & cov from inception_activations
		real_mean = np.load('{}/stat_real_ani.npz'.format(self.real_fid_stat_dir))['mean']
		real_cov = np.load('{}/stat_real_ani.npz'.format(self.real_fid_stat_dir))['cov']
		
		fake_act = np.zeros((test_num, 2048))
		with torch.no_grad():
			for i in tqdm(range(0, test_num, self.batch_size)):
				# randomly sample from animals classes
				label = torch.LongTensor(self.batch_size).random_(0, 398).to(self.device)
				noise = torch.FloatTensor(utils.truncated_normal(self.batch_size*self.z_dim)) \
									.view(self.batch_size, self.z_dim).to(self.device)
				x_sample = self.G(noise, label)

				if i == 0:
					sample_path = os.path.join(self.result_dir, 'test-{}.png'.format(i))
					save_image(utils.denorm(x_sample[:64]), sample_path, nrow=8)

				x_sample = np.transpose(x_sample.cpu().numpy(), (0, 2, 3, 1)) # [NCHW] -> [NHWC]
				fake_act[i:i+self.batch_size] = sess.run(self.activations, {self.images_holder: x_sample})


			fake_mean, fake_cov = np.mean(fake_act, axis=0), np.cov(fake_act, rowvar=False)
			fid = numpy_fid(fake_mean, fake_cov, real_mean, real_cov)

		print('[TinyGAN] Inter-class FID: {:.3f}'.format(fid))
		sess.close()


	def test_inter_fid_big(self):
		test_num = 50000
		class_num = 398

		real_mean = np.load('{}/stat_real_ani.npz'.format(self.real_fid_stat_dir))['mean']
		real_cov = np.load('{}/stat_real_ani.npz'.format(self.real_fid_stat_dir))['cov']
		
		samp_idx = np.random.choice(class_num*5000, test_num, replace=False)
		samp_idx.sort()

		fake_act = []
		for i in tqdm(range(class_num)):
			mask = (i*5000 < samp_idx) * (samp_idx < (i+1)*5000)
			idx = samp_idx[mask] - i*5000
			fake_act.append(np.load('/media/tera/ILSVRC2012/stat_big/act_{}.npy'.format(i))[idx])
		
		fake_act = np.vstack(fake_act)
		
		with torch.no_grad():
			fake_mean, fake_cov = np.mean(fake_act, axis=0), np.cov(fake_act, rowvar=False)
			fid = numpy_fid(fake_mean, fake_cov, real_mean, real_cov)

		print('[BigGAN] Inter-class FID: {:.3f}'.format(fid))


	def test_intra_fid_big_all(self):
		test_num = 5000
		assert test_num % self.batch_size == 0, "test_num mod batch_size != 0"

		config = tf.ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = 0.5
		self.build_fid_graph()
		sess = tf.Session(config=config)
		sess.run(tf.global_variables_initializer())
		sess.graph.finalize()

		fake_act = np.zeros((test_num, 2048))
		fid_scores = []
		with torch.no_grad():
			for c_id in tqdm(range(398)):
				fake_act = np.load('/media/tera/ILSVRC2012/stat_big/act_{}.npy'.format(c_id))
				real_act = np.load(os.path.join(self.real_incep_stat_dir, 'act_{}.npy').format(c_id))
				if self.use_numpy_fid:
					real_mean, real_cov = np.mean(real_act, axis=0), np.cov(real_act, rowvar=False)
					fake_mean, fake_cov = np.mean(fake_act, axis=0), np.cov(fake_act, rowvar=False)
					fid = numpy_fid(fake_mean, fake_cov, real_mean, real_cov)
				else:
					fid = sess.run(self.fid, {self.real_acts: real_act, self.fake_acts: fake_act})
			   
				print('[{}] FID: {:.3f}'.format(c_id, fid))
				fid_scores.append(fid)

		print('[BigGAN] Intra-class FID: {:.3f}, std: {:.3f}'.format(np.mean(fid_scores), np.std(fid_scores)))
		np.save('intra_fid_scores_real_big_deep.npy', fid_scores)


	def test_inception(self):
		test_num = 50000
		assert test_num % self.batch_size == 0, "test_num mod batch_size != 0"
		imgs = np.zeros((test_num, 3, 128, 128))
		with torch.no_grad():
			for i in tqdm(range(0, test_num, self.batch_size)):
				# randomly sample from 1000 classes # [0. 398)
				label = torch.LongTensor(self.batch_size).random_(0, 398).to(self.device)
				noise = torch.FloatTensor(utils.truncated_normal(self.batch_size*self.z_dim))\
									.view(self.batch_size, self.z_dim).to(self.device)
				
				imgs[i:i+self.batch_size] = self.G(noise, label).cpu().numpy() # [NCHW]

		IS_mean, IS_std = get_inception_score(imgs)
		print('IS_mean: {:.2f}, IS_std: {:.2f}'.format(IS_mean, IS_std))


	def test_inception_big(self):
		test_num = 50000
		assert self.train_loader.dataset.num_images >= test_num
		assert test_num % self.batch_size == 0, "test_num mod batch_size != 0"
		imgs = np.zeros((test_num, 3, 128, 128))
		num = 0
		for x_big, _, _ in self.train_loader:
			imgs[num: num+self.batch_size] = x_big
			num += len(x_big)
			if num == test_num: break

		IS_mean, IS_std = get_inception_score(imgs)
		print('IS_mean: {:.2f}, IS_std: {:.2f}'.format(IS_mean, IS_std))


	def test(self):
		nrow = 8
		n_samples = nrow * nrow
		with torch.no_grad():
			for i, (x_real, noise, label) in enumerate(tqdm(self.test_loader)):
				if i == 10: break
				x_real = x_real.to(self.device)
				noise = noise.to(self.device)
				label = label.to(self.device)

				''' test flop '''
				if i==0:
					from thop import profile
					flops, params = profile(self.G, inputs=(noise[0].unsqueeze(0), label[0].unsqueeze(0)))
					print('=======================================================================')
					print('FLOPS: {:.2f} B, Params.: {:.1f} M'.format(flops/10**9, params/10**6))
					print('=======================================================================')				
				# recon
				x_fake = self.G(noise, label)
				comparison = torch.cat([x_real[:nrow], x_fake[:nrow].float()])
				sample_path = os.path.join(self.result_dir, '{}-rec.png'.format(i+1))
				save_image(utils.denorm(comparison.cpu()), sample_path)

				# sample
				noise2 = torch.FloatTensor(utils.truncated_normal(n_samples*self.z_dim)) \
									.view(n_samples, self.z_dim).to(self.device)
				label = label[:nrow].repeat(nrow)
				x_sample = self.G(noise2, label)
				sample_path = os.path.join(self.result_dir, '{}-sample.png'.format(i+1))
				save_image(utils.denorm(x_sample), sample_path, nrow=nrow)


	def test_interpolate(self):
		with torch.no_grad():
			for i, (x_real, noise, label) in enumerate(self.test_loader):
				if i == 10: break
				x_real = x_real.squeeze()
				noise = noise.squeeze().to(self.device)
				label = label.squeeze().to(self.device)
				
				y_emb = self.G.embeding(label) 

				interval = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
				y_emb_list = torch.zeros(len(interval), self.c_dim, dtype=torch.to(self.device))
				noise_list = torch.zeros(len(interval), self.z_dim, dtype=torch.to(self.device))

				for j, alpha in enumerate(interval): # from label0 to label1
					y_emb_list[j] = y_emb[0]*(1-alpha) + y_emb[1]*alpha
					noise_list[j] = noise[0]*(1-alpha) + noise[1]*alpha

				noise0 = noise[0].repeat(len(interval)).view(len(interval), -1)
				noise1 = noise[1].repeat(len(interval)).view(len(interval), -1)
				
				# recon # interpolate y
				x_fakey0 = self.G.interpolate(noise0, y_emb_list).float().cpu()
				x_fakey1 = self.G.interpolate(noise1, y_emb_list).float().cpu()
				x_fakez0 = self.G(noise_list, label[0].repeat(len(interval))).float().cpu()
				x_fakez1 = self.G(noise_list, label[1].repeat(len(interval))).float().cpu()

				real_list = torch.zeros_like(x_fakey0) # padding
				real_list[0] = x_real[0]
				real_list[-1] = x_real[1]

				# [len(interval)*5, 3, 128, 128]
				comparison = torch.cat([real_list, x_fakey0, x_fakey1, x_fakez0, x_fakez1])

				sample_path = os.path.join(self.result_dir, '{}-rec.png'.format(i+1))
				save_image(utils.denorm(comparison), sample_path, nrow=len(interval))	   
