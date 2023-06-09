<template>
    <div class="container">
        <div class="">
            <div class="">
                <h1>Books</h1>
                <hr><br><br>
                <button type="button" class="">Add Book</button>
                <br><br>
                <table class="">
                    <thead>
                        <tr>
                            <th scope="col">Title</th>
                            <th scope="col">Author</th>
                            <th scope="col">Read?</th>
                            <th></th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr v-for="(book, index) in books" :key="index">
                            <td>{{ book.title }}</td>
                            <td>{{ book.author }}</td>
                            <td>
                                <span v-if="book.read">Yes</span>
                                <span v-else>No</span>
                            </td>
                            <td>
                                <div class="" role="group">
                                    <button type="button" class="">Update</button>
                                    <button type="button" class="">Delete</button>
                                </div>
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</template>

<script>
import axios from 'axios';

export default {
  data() {
    return {
      books: [],
    };
  },
  methods: {
    getBooks() {
      const path = 'http://localhost:5001/books';
      axios.get(path)
        .then((res) => {
          this.books = res.data.books;
        })
        .catch((error) => {
          console.error(error);
        });
    },
  },
  created() {
    this.getBooks();
  },
};
</script>