type UserBaseType = {
  _id?: string;
  email: string;
  name: string;
  username: string;
  phoneNumber: PhoneNumberType;
  location?: LocationType;
  profilePicture: string;
  socialLinks?: {
    facebook?: URLType | string;
    twitter?: URLType | string;
    instagram?: URLType | string;
    linkedin?: URLType | string;
    tiktok?: URLType | string;
    youtube?: URLType | string;
    website?: URLType | string;
    [key: string]: URLType | string | undefined;
  };
  verificationType?: "email" | "phone";
  description?: string;
  preferences?: Array<string>;
  blurhash?: string;
};

type SchoolType = UserBaseType & {
  userType: "school"; // sub-school school-staff
  last?: null;
  schoolType:
    | "private"
    | "language"
    | "university"
    | "formation"
    | "public"
    | "support"
    | "private-university"
    | "preschool";
  DOB?: null;
  gender?: null;
};

type ParentType = UserBaseType & {};

type NonSchoolType = UserBaseType & {
  userType: "student" | "teacher" | "parent" | "admin";
  last: string;
  schoolType?: null;
  DOB?: number | null;
  gender?: "male" | "female" | null;
};

export type UserType = SchoolType | NonSchoolType;

interface LocationType {
  wilaya: string;
  commune: string;
  coordinates: {
    lang: number;
    lat: number;
  };
  fullLocation: string;
}

type PhoneNumberType = string | null;

type URLType = string | null;
