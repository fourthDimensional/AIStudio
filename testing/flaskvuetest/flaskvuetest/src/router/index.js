import { createRouter, createWebHistory } from 'vue-router'
import Books from '../components/books.vue'
import Ping from '../components/ping.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/books',
      name: 'books',
      component: Books,
    },
    {
      path: '/ping',
      name: 'ping',
      component: Ping
    },
  ]
})

export default router
