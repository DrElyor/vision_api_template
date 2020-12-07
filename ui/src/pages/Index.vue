<template>
  <q-page>
    <div class="row">
      <div class="q-pa-md q-gutter-sm">
        <q-breadcrumbs class="text-brown">
          <template v-slot:separator>
            <q-icon size="1.5em" name="chevron_right" color="primary" />
          </template>

          <q-breadcrumbs-el label="Home" icon="home" />
          <q-breadcrumbs-el label="Vision API" icon="widgets" />
          <q-breadcrumbs-el label="Demo Api" icon="camera" />
        </q-breadcrumbs>
      </div>
    </div>

    <div class="q-pa-md row items-start q-gutter-md flex flex-center">
      <q-card class="my-card" flat bordered>
        <q-card-section horizontal>
          <q-img class="col-5" src="https://loremflickr.com/640/360" />

          <q-card-section
            ><div class="row">{{ lorem }}</div>
            <div class="q-pa-md row items-start q-gutter-md flex flex-center">
              <q-uploader
                url="http://localhost:8000/api/demo"
                style="max-width: 50vw; height: 20%"
                multiple
                field-name="file"
                @uploaded="showResponse"
              />
            </div>
          </q-card-section>
        </q-card-section>
      </q-card>
    </div>

    <div class="row flex-center">
      <div class="q-pa-md q-gutter-sm" style="width: 70vw">
        <q-img :src="url" spinner-color="white" />
      </div>
    </div>

    
  </q-page>
</template>

<script>
export default {
  name: "PageIndex",
  data: function () {
    return {
      url: "",
      lorem:
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
    };
  },
  methods: {
    showResponse(info) {
      this.url = JSON.parse(info["xhr"]["response"]).encoded_img;
    },
  },
};
</script>

<style lang="sass" scoped>
.my-card
  width: 100%
  max-width: 50vw
</style>