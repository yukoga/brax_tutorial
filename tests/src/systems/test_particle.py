import sys
sys.path.append('./src/')

import pytest
import numpy as np
from pyphysics.systems.particle import Particle, Particles


@pytest.fixture
def particles():
    return Particles([
        Particle(np.array([0., 1.]), np.array([1., 1.]), 1.0),
        Particle(np.array([0., 2.]), np.array([2., 2.]), 2.0)
    ])


def test_init_particle(particles):
    assert all(particles[0].q == np.array([0., 1.]))
    assert all(particles[0].p == np.array([1., 1.]))


def test_update_particle(particles):
    particles[0].q = np.array([2., 2.])
    particles[0].p = np.array([1., 0.])
    assert all(particles[0].q == np.array([2., 2.]))
    assert all(particles[0].p == np.array([1., 0.]))


def test_particle_velocity(particles):
    assert all(particles[0].v == np.array([1., 1.])/1.0)


def test_kinetic_energy(particles):
    assert particles[0].ke == 2.0


def test_init_multiple_particles(particles):
    assert all(particles[0].q == np.array([0., 1.]))
    assert all(particles[0].p == np.array([1., 1.]))
    assert all(particles[1].q == np.array([0., 2.]))
    assert all(particles[1].p == np.array([2., 2.]))


def test_add_particle_to_particles(particles):
    particles.add(Particle(np.array([0., 3.]), np.array([3., 3.]), 3.0))

    assert all(particles[2].q == np.array([0., 3.]))
    assert all(particles[2].p == np.array([3., 3.]))


def test_iterate_particles(particles):
    q_list = [np.array([0., 1.]), np.array([0., 2.])]
    p_list = [np.array([1., 1.]), np.array([2., 2.])]
    m_list = [1.0, 2.0]

    for i, particle in enumerate(particles):
        assert particle.m == m_list[i]
        assert all(particle.q == q_list[i])
        assert all(particle.p == p_list[i])
        assert all(particle.v == p_list[i] / m_list[i])
